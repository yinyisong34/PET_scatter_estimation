import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from basis_functions import convert_PDF_to_bin_probability



def system_matrix_A(U_E, S_E, dE):
    # U_E and and S_E are the PDF of non-scatter photons and scattered photons repespectively.
    U_E_bin = convert_PDF_to_bin_probability(U_E, dE)
    S_E_bin = convert_PDF_to_bin_probability(S_E, dE)
    A_bin = np.column_stack((U_E_bin, S_E_bin))
    return A_bin


def prompt_histogram_1D_y1_and_y2 (total_histogram):
    y1_E = np.sum(total_histogram, axis = 1)
    y2_E = np.sum(total_histogram, axis = 0)
    return y1_E, y2_E

def Cy_setup_initialization(y1_E, y2_E):
    # FWHM = 3 energy bins
    FWHM_bins = 3

    # Convert FWHM to Gaussian sigma
    gaussian_sigma = FWHM_bins / math.sqrt(8 * math.log(2))

    # y is your measured 1D marginal prompt energy spectrum
    # Smooth the spectrum
    y1_smooth = gaussian_filter1d(y1_E.astype(float), sigma=gaussian_sigma)
    y2_smooth = gaussian_filter1d(y2_E.astype(float), sigma=gaussian_sigma)

    # For Poisson data: variance ≈ expected counts
    # So covariance matrix is diagonal with smoothed counts
    eps = 1e-8

    Cy1 = np.maximum(y1_smooth, eps) # Cy1_diag
    Cy2 = np.maximum(y2_smooth, eps) # Cy2_diag 

    return Cy1, Cy2

def L(p, yl, A_bin, Cy_diag):
    residual = yl - A_bin @ p
    L = np.sum((residual**2)/Cy_diag)
    return L


def iteration (L, p0, A_bin, y1_E, y2_E, Cy1, Cy2):
    eps = 1e-8

    result1 = minimize(
        L,
        x0=p0,
        args=(y1_E, A_bin, Cy1),
        bounds=[(0, None)] * A_bin.shape[1]
    )

    p_hat_1 = result1.x

    result2 = minimize(
        L,
        x0=p0,
        args=(y2_E, A_bin, Cy2),
        bounds=[(0, None)] * A_bin.shape[1]
    )

    p_hat_2 = result2.x

    Cy1 = np.maximum(A_bin @ p_hat_1, eps)
    Cy2 = np.maximum(A_bin @ p_hat_2, eps)
    return p_hat_1, p_hat_2, Cy1, Cy2






def initialize_coefficients_for_NEGML(L, A_bin, y1_E, y2_E):
    Cy1, Cy2 = Cy_setup_initialization(y1_E, y2_E)

    p0 = np.ones(A_bin.shape[1])   # initial guess

    for i in range(3):
        p_hat_1, p_hat_2, Cy1, Cy2 = iteration (L, p0, A_bin, y1_E, y2_E, Cy1, Cy2)

    p1_sum = np.sum(p_hat_1)
    p2_sum = np.sum(p_hat_2)
    N = np.sum(y1_E)

    theta = N * np.array([p_hat_1[0]/p1_sum*p_hat_2[0]/p2_sum, 
                        p_hat_1[0]/p1_sum*p_hat_2[1]/p2_sum, 
                        p_hat_1[1]/p1_sum*p_hat_2[0]/p2_sum, 
                        p_hat_1[1]/p1_sum*p_hat_2[1]/p2_sum ])
    return theta




def negml_4param(y_2D, basis, r_hat_2D=None, theta_init=None,
                 n_iter=100, eps=1e-8):
    """
    y_2D: measured 2D energy histogram
    basis: shape (4, nE, nE), containing UU, US, SU, SS
    r_hat_2D: estimated random 2D histogram, same shape as y_2D
    theta_init: initial coefficients, length 4 in this case
    """

    y_2D = np.asarray(y_2D, dtype=float) # shape (10, )
    basis = np.asarray(basis, dtype=float) # shape (4, 10, 10)

    if r_hat_2D is None:
        r_hat_2D = np.zeros_like(y_2D)
    else:
        r_hat_2D = np.asarray(r_hat_2D, dtype=float)

    if theta_init is None:
        sigma = np.ones(4) # shape (4, )
    else:
        sigma = np.asarray(theta_init, dtype=float).copy()

    for it in range(n_iter):

        for k in range(4):  # sequential update, one parameter at a time

            # current estimated non-random spectrum
            n_hat = (sigma[0] * basis[0] # () * (10,10)
                    + sigma[1] * basis[1]
                    + sigma[2] * basis[2]
                    + sigma[3] * basis[3]
                    )

            # current estimated prompt spectrum
            y_hat = n_hat + r_hat_2D # shape (10,10)
            y_hat = np.maximum(y_hat, eps)

            c_k = basis[k] # dn/dsigma = P(Ei),P(Ej), shape(10,10) 

            numerator = np.sum(c_k * (y_2D - y_hat) / y_hat)
            denominator = np.sum((c_k ** 2) / y_hat)

            update = numerator / np.maximum(denominator, 1e-8)

            sigma[k] = sigma[k] + update

            # non-negativity constraint
            sigma[k] = max(sigma[k], 0)

    # n_hat = np.sum(sigma[:, None, None] * basis, axis=0)
    # y_hat = n_hat + r_hat_2D

    return sigma #, n_hat, y_hat


def multi_micro_negml_4param(sino_shape, Y, basis, theta, n_iter=100):
    sigma_negml = np.zeros((4,) + sino_shape)

    fraction = theta / np.sum(theta)

    for idx in np.ndindex(sino_shape):
        y_2D_bin = Y[(slice(None), slice(None)) + idx]
        N_bin = np.sum(y_2D_bin)

        if N_bin > 0:
            theta_init_bin = fraction * N_bin

            sigma_negml[(slice(None),) + idx] = negml_4param(
                y_2D=y_2D_bin,
                basis=basis,
                theta_init=theta_init_bin,
                n_iter=n_iter
            )

    return sigma_negml