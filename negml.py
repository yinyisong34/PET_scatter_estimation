"""NEGML coefficient-estimation and initialisation utilities.

This module contains the one-dimensional weighted fitting used to initialise
the four EBSE coefficients, the four-parameter NEGML update for a joint
energy histogram, and a wrapper for applying a fixed basis throughout a
sinogram."""

import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from basis_functions import convert_PDF_to_bin_probability



def system_matrix_A(U_E, S_E, dE):
    """Construct the 1D two-component system matrix used for initialisation.
    
    Parameters
    ----------
    U_E, S_E : array-like
        Sampled non-scatter and scatter energy PDFs.
    dE : float
        Energy-bin width in keV.
    
    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_energy, 2)`` whose columns contain the discrete
        non-scatter and scatter bin probabilities.
    """

    # U_E and and S_E are the PDF of non-scatter photons and scattered photons repespectively.
    U_E_bin = convert_PDF_to_bin_probability(U_E, dE)
    S_E_bin = convert_PDF_to_bin_probability(S_E, dE)
    A_bin = np.column_stack((U_E_bin, S_E_bin))
    return A_bin


def prompt_histogram_1D_y1_and_y2 (total_histogram):
    """Compute the two marginal energy histograms of a joint prompt spectrum.
    
    Parameters
    ----------
    total_histogram : numpy.ndarray
        Joint energy histogram with dimensions ``(E1, E2)``.
    
    Returns
    -------
    y1_E : numpy.ndarray
        First-photon marginal obtained by summing over E2.
    y2_E : numpy.ndarray
        Second-photon marginal obtained by summing over E1.
    """
    y1_E = np.sum(total_histogram, axis = 1)
    y2_E = np.sum(total_histogram, axis = 0)
    return y1_E, y2_E

def Cy_setup_initialization(y1_E, y2_E):
    """Initialise diagonal variance estimates for the two marginal spectra.
    
    Each marginal prompt spectrum is smoothed with a Gaussian whose FWHM is
    three energy bins. The smoothed counts are then used as Poisson-like
    variance estimates for the weighted initialisation fit.
    
    Parameters
    ----------
    y1_E, y2_E : array-like
        First- and second-photon marginal prompt-energy histograms.
    
    Returns
    -------
    Cy1, Cy2 : numpy.ndarray
        Diagonal variance vectors for the two marginal fits.
    """
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
    """Evaluate the weighted least-squares objective used for initialisation.
    
    Parameters
    ----------
    p : array-like
        Current two-component coefficient vector.
    yl : array-like
        Observed one-dimensional marginal spectrum.
    A_bin : numpy.ndarray
        System matrix containing non-scatter and scatter bin probabilities.
    Cy_diag : array-like
        Diagonal variance estimate for each energy bin.
    
    Returns
    -------
    float
        Weighted sum of squared residuals.
    """
    residual = yl - A_bin @ p
    L = np.sum((residual**2)/Cy_diag)
    return L


def iteration (L, p0, A_bin, y1_E, y2_E, Cy1, Cy2):
    """Perform one variance-update cycle for both marginal initialisation fits.
    
    Non-negative weighted least-squares fits are performed independently for
    the first and second marginal spectra using the current variance estimates.
    The fitted model values are then used to update those variances.
    
    Parameters
    ----------
    L : callable
        Objective function used by ``scipy.optimize.minimize``.
    p0 : array-like
        Initial coefficient vector for each marginal fit.
    A_bin : numpy.ndarray
        Two-component system matrix.
    y1_E, y2_E : array-like
        Marginal prompt-energy histograms.
    Cy1, Cy2 : array-like
        Current diagonal variance estimates.
    
    Returns
    -------
    p_hat_1, p_hat_2 : numpy.ndarray
        Fitted non-scatter/scatter coefficients for the two marginals.
    Cy1, Cy2 : numpy.ndarray
        Updated diagonal variance estimates.
    """
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
    """Estimate a four-component starting vector for the NEGML fit.
    
    The two marginal spectra are fitted as non-scatter/scatter mixtures. Their
    normalised fitted fractions are combined under a factorised assumption and
    scaled by the total number of events to initialise the UU, US, SU, and SS
    coefficients.
    
    Parameters
    ----------
    L : callable
        Weighted objective function used by the marginal fits.
    A_bin : numpy.ndarray
        Two-column non-scatter/scatter system matrix.
    y1_E, y2_E : array-like
        First- and second-photon marginal prompt-energy histograms.
    
    Returns
    -------
    numpy.ndarray
        Initial four-component coefficient vector ordered as
        ``[UU, US, SU, SS]``.
    """
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
    """Fit four non-negative EBSE coefficients to a joint energy histogram.
    
    The current model is formed as a weighted sum of the four supplied basis
    images. Each coefficient is updated sequentially using the implemented
    NEGML coordinate update and constrained to remain non-negative.
    
    Parameters
    ----------
    y_2D : array-like
        Measured joint prompt-energy histogram with shape
        ``(n_energy, n_energy)``.
    basis : array-like
        Four basis images with shape ``(4, n_energy, n_energy)``, conventionally
        ordered as UU, US, SU, and SS.
    r_hat_2D : array-like, optional
        Estimated random-coincidence joint-energy histogram. If omitted, a zero
        array is used.
    theta_init : array-like, optional
        Initial four-component coefficient vector. If omitted, all coefficients
        start at one.
    n_iter : int, optional
        Number of complete sequential-update iterations. Default is 100.
    eps : float, optional
        Numerical floor used to avoid division by zero. Default is ``1e-8``.
    
    Returns
    -------
    numpy.ndarray
        Fitted non-negative coefficient vector of length four.
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
    """Apply a fixed four-component basis independently to every sinogram bin.
    
    The relative fractions of the supplied global initial coefficients are
    scaled by the total counts in each local joint-energy histogram before
    calling ``negml_4param``.
    
    Parameters
    ----------
    sino_shape : tuple
        Spatial sinogram shape.
    Y : numpy.ndarray
        Energy-resolved prompt data with shape
        ``(E1, E2) + sino_shape``.
    basis : numpy.ndarray
        Fixed four-component energy basis.
    theta : array-like
        Global initial coefficient vector whose fractions are reused locally.
    n_iter : int, optional
        Number of NEGML iterations per spatial bin. Default is 100.
    
    Returns
    -------
    numpy.ndarray
        Fitted coefficient array with shape ``(4,) + sino_shape``.
    """
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