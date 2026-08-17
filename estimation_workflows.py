import numpy as np
import basis_functions
import negml


def multi_micro_negml_4param_avg_z_axis(sino_shape, Y, basis, theta, n_iter=100):
    
    sigma_negml = np.zeros((4,) + sino_shape)

    fraction = theta / np.sum(theta)

    # Y shape assumed:
    # (E1, E2, segment, axial, view, tangential)
    # Average over axial direction
    Y_avg = np.mean(Y, axis=3)

    n_segment, n_axial, n_view, n_tangential = sino_shape

    for segment in range(n_segment):
        for view in range(n_view):
            for tangential in range(n_tangential):

                # This spectrum already represents the average
                # over all axial positions
                y_2D_bin = Y_avg[:, :, segment, view, tangential]

                N_bin = np.sum(y_2D_bin)

                if N_bin > 0:
                    theta_init_bin = fraction * N_bin

                    theta_fit = negml.negml_4param(
                        y_2D=y_2D_bin,
                        basis=basis,
                        theta_init=theta_init_bin,
                        n_iter=n_iter
                    )

                    # Put the same fitted result into ALL axial positions
                    sigma_negml[
                        :, segment, :, view, tangential
                    ] = theta_fit[:, None]

    return sigma_negml

def procedure_with_all_z_axis_no_mask(total_histogram, PDF_nonscatter, PDF_scatter, sino_shape, sinogram_with_energy_bin_info, dE = 20):
    U_E = np.array(PDF_nonscatter)
    S_E = np.array(PDF_scatter)
    A_bin  = negml.system_matrix_A(U_E, S_E, 20)

    y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram)

    theta = negml.initialize_coefficients_for_NEGML(negml.L, A_bin, y1_E, y2_E)
    print(theta)


    basis = basis_functions.derive_shared_1d_factorised_basis(U_E, S_E, dE)


    theta = negml.negml_4param(
    y_2D=total_histogram,
    basis=basis,
    r_hat_2D=None,        # or your random estimate if you have one
    theta_init = theta,
    n_iter=100
    )
    print(theta)

    sigma_negml = negml.multi_micro_negml_4param(sino_shape, sinogram_with_energy_bin_info, basis, theta, n_iter=100)
  
    return theta, sigma_negml


def procedure_with_avg_z_axis_no_mask(total_histogram, PDF_nonscatter, PDF_scatter, sino_shape, sinogram_with_energy_bin_info,dE = 20):
    U_E = np.array(PDF_nonscatter)
    S_E = np.array(PDF_scatter)
    A_bin  = negml.system_matrix_A(U_E, S_E, 20)

    y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram)

    theta = negml.initialize_coefficients_for_NEGML(negml.L, A_bin, y1_E, y2_E)
    print(theta)


    basis = basis_functions.derive_shared_1d_factorised_basis(U_E, S_E, dE)


    theta = negml.negml_4param(
    y_2D=total_histogram,
    basis=basis,
    r_hat_2D=None,        # or your random estimate if you have one
    theta_init = theta,
    n_iter=100
    )
    print(theta)

    sigma_negml = negml.multi_micro_negml_4param_avg_z_axis(sino_shape, sinogram_with_energy_bin_info, basis, theta, n_iter=100)
  
    return theta, sigma_negml



def single_micro_negml_4param_avg_z_axis(i, pixel_width, j, pixel_length, Y_avg, basis, theta_init_bin, sigma_negml, n_iter=100):

    for m in range(pixel_width):
        for n in range(pixel_length):
            
            y_2D_bin = Y_avg[:, :, 0, i * pixel_width + m, j * pixel_length + n]
            N_bin = np.sum(y_2D_bin)

            if N_bin > 0:

                sigma_result = negml.negml_4param(
                    y_2D=y_2D_bin,
                    basis=basis,
                    theta_init=theta_init_bin,
                    n_iter=n_iter
                )

                sigma_negml[
                    :,
                    0,
                    :,
                    i * pixel_width + m,
                    j * pixel_length + n
                ] = sigma_result[:, None]

    return sigma_negml



