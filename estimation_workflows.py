import numpy as np
import energy_pdfs
import basis_functions
import negml
import spatial_regions


def procedure_with_all_z_axis_no_mask(total_histogram, PDF_nonscatter, PDF_scatter, sino_shape, sinogram_with_energy_bin_info, dE = 20):
    U_E = np.array(PDF_nonscatter)
    S_E = np.array(PDF_scatter)
    A_bin  = negml.system_matrix_A(U_E, S_E, 20)

    y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram)

    theta = negml.initialize_coefficients_for_NEGML(negml.L, A_bin, y1_E, y2_E)


    basis = basis_functions.derive_shared_1d_factorised_basis(U_E, S_E, dE)


    theta = negml.negml_4param(
    y_2D=total_histogram,
    basis=basis,
    r_hat_2D=None,        # or your random estimate if you have one
    theta_init = theta,
    n_iter=100
    )

    sigma_negml = negml.multi_micro_negml_4param(sino_shape, sinogram_with_energy_bin_info, basis, theta, n_iter=100)
  
    return sigma_negml


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

    sigma_negml = multi_micro_negml_4param_avg_z_axis(sino_shape, sinogram_with_energy_bin_info, basis, theta, n_iter=100)
  
    return sigma_negml



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


energies = list(range(410, 591, 20))

def procedure_with_avg_z_axis_basis_for_each_mask(pixel_width, pixel_length, Func_derive_basis, PDF_nonscatter_true_or_predicted, PDF_nonscatter, PDF_scatter_true_or_predicted, sino_shape, sinogram_with_energy_bin_info,  c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info, energies = energies, dE = 20):

    Y_avg = np.mean(sinogram_with_energy_bin_info, axis=3)
    
    sigma_negml = np.zeros((4,) + sino_shape) #(4, 1, 12, 32, 55)

    pixel_width = int(pixel_width) 
    pixel_length = int(pixel_length)    
    for i in range(int(32/pixel_width)):
        for j in range(int(55/pixel_length)):
            
            mask = np.zeros((32,55))
            for m in range(pixel_width):
                for n in range(pixel_length):
                    mask += spatial_regions.creat_mask_at( i * pixel_width + m, j * pixel_length + n)

            mask = mask.astype(bool)


            # true U true S
            if (PDF_nonscatter_true_or_predicted == "true") and (PDF_scatter_true_or_predicted == "true"):
                nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
                total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
                y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)
                                
            # predicted U true S
            elif (PDF_nonscatter_true_or_predicted == "predicted") and (PDF_scatter_true_or_predicted == "true"):
                nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
                nonscatter1_in_pdf = PDF_nonscatter
                nonscatter2_in_pdf = PDF_nonscatter
                total_histogram_inside_mask = total_histogram_inside_mask(mask, Y_avg)
                y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)
                                
            # true U predicted S   
            elif (PDF_nonscatter_true_or_predicted == "true") and (PDF_scatter_true_or_predicted == "predicted"):
                nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
                total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
                y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)                         
                scatter1_in_pdf = energy_pdfs.estimate_scatter_PDF(y1_E, nonscatter1_in_pdf, energies)
                scatter2_in_pdf = energy_pdfs.estimate_scatter_PDF(y2_E, nonscatter2_in_pdf, energies)

            # predicted U predicted S 
            elif (PDF_nonscatter_true_or_predicted == "predicted") and (PDF_scatter_true_or_predicted == "predicted"):
                nonscatter1_in_pdf = PDF_nonscatter
                nonscatter2_in_pdf = PDF_nonscatter
                total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
                y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)                        
                scatter1_in_pdf = energy_pdfs.estimate_scatter_PDF(y1_E, nonscatter1_in_pdf, energies)
                scatter2_in_pdf = energy_pdfs.estimate_scatter_PDF(y2_E, nonscatter2_in_pdf, energies)

            U_E = np.array(nonscatter1_in_pdf)
            S_E = np.array(scatter1_in_pdf)
            A_bin  = negml.system_matrix_A(U_E, S_E, dE)

            theta = negml.initialize_coefficients_for_NEGML(negml.L, A_bin, y1_E, y2_E)

            if (Func_derive_basis == basis_functions.derive_ground_truth_2d_basis):
                basis = basis_functions.derive_ground_truth_2d_basis(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
            if (Func_derive_basis == basis_functions.derive_separate_1d_factorised_basis):
                basis = basis_functions.derive_separate_1d_factorised_basis(nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf, dE)
            elif (Func_derive_basis == basis_functions.derive_shared_1d_factorised_basis):
                basis = basis_functions.derive_shared_1d_factorised_basis(nonscatter1_in_pdf, scatter1_in_pdf, dE)

            sigma_negml = single_micro_negml_4param_avg_z_axis(i, pixel_width, j, pixel_length, Y_avg, basis, theta, sigma_negml, n_iter=100)
  
    return sigma_negml

def procedure_with_avg_z_axis_basis_for_single_pixel(mask, Func_derive_basis, PDF_nonscatter_true_or_predicted, PDF_nonscatter, PDF_scatter_true_or_predicted, sino_shape, sinogram_with_energy_bin_info,  c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info, energies = energies, dE = 20):

    Y_avg = np.mean(sinogram_with_energy_bin_info, axis=3)

    sigma_negml = np.zeros((4,) + sino_shape) #(4, 1, 12, 32, 55)

    # true U true S
    if (PDF_nonscatter_true_or_predicted == "true") and (PDF_scatter_true_or_predicted == "true"):
        nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
        total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
        y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)
                                
    # predicted U true S
    elif (PDF_nonscatter_true_or_predicted == "predicted") and (PDF_scatter_true_or_predicted == "true"):
        nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
        nonscatter1_in_pdf = PDF_nonscatter
        nonscatter2_in_pdf = PDF_nonscatter
        total_histogram_inside_mask = total_histogram_inside_mask(mask, Y_avg)
        y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)
                                
    # true U predicted S   
    elif (PDF_nonscatter_true_or_predicted == "true") and (PDF_scatter_true_or_predicted == "predicted"):
        nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf = spatial_regions.true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
        total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
        y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)                         
        scatter1_in_pdf = energy_pdfs.estimate_scatter_PDF(y1_E, nonscatter1_in_pdf, energies)
        scatter2_in_pdf = energy_pdfs.estimate_scatter_PDF(y2_E, nonscatter2_in_pdf, energies)

    # predicted U predicted S 
    elif (PDF_nonscatter_true_or_predicted == "predicted") and (PDF_scatter_true_or_predicted == "predicted"):
        nonscatter1_in_pdf = PDF_nonscatter
        nonscatter2_in_pdf = PDF_nonscatter
        total_histogram_inside_mask = spatial_regions.total_histogram_inside_mask(mask, Y_avg)
        y1_E, y2_E = negml.prompt_histogram_1D_y1_and_y2 (total_histogram_inside_mask)                        
        scatter1_in_pdf = energy_pdfs.estimate_scatter_PDF(y1_E, nonscatter1_in_pdf, energies)
        scatter2_in_pdf = energy_pdfs.estimate_scatter_PDF(y2_E, nonscatter2_in_pdf, energies)

    U_E = np.array(nonscatter1_in_pdf)
    S_E = np.array(scatter1_in_pdf)
    A_bin  = negml.system_matrix_A(U_E, S_E, dE)

    theta = negml.initialize_coefficients_for_NEGML(negml.L, A_bin, y1_E, y2_E)

    if (Func_derive_basis == basis_functions.derive_ground_truth_2d_basis):
        basis = basis_functions.derive_ground_truth_2d_basis(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info)
    if (Func_derive_basis == basis_functions.derive_separate_1d_factorised_basis):
        basis = basis_functions.derive_separate_1d_factorised_basis(nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf, dE)
    elif (Func_derive_basis == basis_functions.derive_shared_1d_factorised_basis):
        basis = basis_functions.derive_shared_1d_factorised_basis(nonscatter1_in_pdf, scatter1_in_pdf, dE)

    sigma_negml = negml.negml_4param(total_histogram_inside_mask, basis)
  
    return sigma_negml