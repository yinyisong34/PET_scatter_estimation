# SPDX-License-Identifier: Apache-2.0
"""High-level workflows for applying NEGML to global and regional EBSE models.

This module coordinates energy-PDF selection, basis-function construction,
axial averaging, spatial masking, NEGML initialisation, and repeated
coefficient fitting. The low-level NEGML update itself is implemented in
``negml.py``."""

import numpy as np
import energy_pdfs
import basis_functions
import negml
import spatial_regions


def procedure_with_all_z_axis_no_mask(total_histogram, PDF_nonscatter, PDF_scatter, sino_shape, sinogram_with_energy_bin_info, dE = 20):
    """Run the global shared-basis NEGML workflow without axial averaging.
    
    A single pair of supplied U(E) and S(E) PDFs is used to construct a shared
    1D factorised basis. Initial coefficients are estimated from the global
    marginal spectra and refined on the global 2D histogram before the same
    basis is fitted independently to every full sinogram bin.
    
    Parameters
    ----------
    total_histogram : numpy.ndarray
        Global joint-energy histogram with shape ``(n_energy, n_energy)``.
    PDF_nonscatter, PDF_scatter : array-like
        Global non-scatter and scatter energy PDFs.
    sino_shape : tuple
        Spatial sinogram shape.
    sinogram_with_energy_bin_info : numpy.ndarray
        Energy-resolved prompt data with shape
        ``(n_energy, n_energy) + sino_shape``.
    dE : float, optional
        Energy-bin width in keV. Default is 20.
    
    Returns
    -------
    numpy.ndarray
        NEGML coefficient array with shape ``(4,) + sino_shape``.
    """
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
    """Apply a fixed basis after averaging the energy data over the axial axis.
    
    The energy-resolved data are averaged over axis 3, assumed to be the axial
    dimension. One four-parameter NEGML fit is performed for each
    segment--view--tangential location, and the fitted coefficients are copied
    to all axial positions in the output array.
    
    Parameters
    ----------
    sino_shape : tuple
        Spatial sinogram shape ``(segment, axial, view, tangential)``.
    Y : numpy.ndarray
        Energy-resolved prompt data with shape
        ``(E1, E2, segment, axial, view, tangential)``.
    basis : numpy.ndarray
        Four-component energy basis with shape ``(4, n_energy, n_energy)``.
    theta : array-like
        Global initial coefficient vector. Its relative fractions are scaled to
        the local event total before each fit.
    n_iter : int, optional
        Number of NEGML iterations. Default is 100.
    
    Returns
    -------
    numpy.ndarray
        Coefficient array with shape ``(4,) + sino_shape``.
    """
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
    """Run the global shared-basis NEGML workflow with axial averaging.
    
    This follows the same global PDF, basis, and coefficient-initialisation
    steps as ``procedure_with_all_z_axis_no_mask``, but the final spatial fits
    are performed on axially averaged energy histograms.
    
    Parameters
    ----------
    total_histogram : numpy.ndarray
        Global ``(E1, E2)`` prompt-energy histogram.
    PDF_nonscatter, PDF_scatter : array-like
        Global non-scatter and scatter energy PDFs.
    sino_shape : tuple
        Spatial sinogram shape ``(segment, axial, view, tangential)``.
    sinogram_with_energy_bin_info : numpy.ndarray
        Full energy-resolved prompt sinogram.
    dE : float, optional
        Energy-bin width in keV. Default is 20.
    
    Returns
    -------
    numpy.ndarray
        NEGML coefficient array with shape ``(4,) + sino_shape``.
    """
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
    """Fit all pixels belonging to one rectangular spatial region.
    
    Each non-empty view--tangential pixel in the selected region is fitted using
    the same region-specific basis and supplied initial coefficient vector.
    The energy data are assumed to have already been averaged over the axial
    dimension, and each fitted result is copied across all axial positions.
    
    Parameters
    ----------
    i, j : int
        Region indices in the view and tangential directions.
    pixel_width, pixel_length : int
        Width and length of the rectangular region in pixels.
    Y_avg : numpy.ndarray
        Axially averaged energy-resolved data with shape
        ``(E1, E2, segment, view, tangential)``.
    basis : numpy.ndarray
        Region-specific four-component energy basis.
    theta_init_bin : array-like
        Initial four NEGML coefficients used for pixels in this region.
    sigma_negml : numpy.ndarray
        Output coefficient array to update in place.
    n_iter : int, optional
        Number of NEGML iterations. Default is 100.
    
    Returns
    -------
    numpy.ndarray
        The updated ``sigma_negml`` coefficient array.
    """
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
    """Run regional NEGML using one basis for each rectangular spatial mask.
    
    The view--tangential plane is partitioned into regions of size
    ``pixel_width`` by ``pixel_length``. For each region, the requested
    ground-truth or predicted U/S PDFs are obtained, the selected basis
    constructor is applied, initial coefficients are estimated from the local
    marginal spectra, and NEGML is fitted to the pixels in that region.
    
    Parameters
    ----------
    pixel_width, pixel_length : int
        Dimensions of each rectangular region in the 32-by-55 view--tangential
        plane.
    Func_derive_basis : callable
        Basis constructor. Supported workflows use
        ``derive_shared_1d_factorised_basis``,
        ``derive_separate_1d_factorised_basis``, or
        ``derive_ground_truth_2d_basis``.
    PDF_nonscatter_true_or_predicted : {"true", "predicted"}
        Select whether the regional non-scatter PDF is simulation-derived or
        replaced by ``PDF_nonscatter``.
    PDF_nonscatter : array-like
        Predicted global non-scatter PDF used when the non-scatter option is
        ``"predicted"``.
    PDF_scatter_true_or_predicted : {"true", "predicted"}
        Select whether the scatter PDF is simulation-derived or estimated from
        the regional marginal spectrum.
    sino_shape : tuple
        Spatial sinogram shape.
    sinogram_with_energy_bin_info : numpy.ndarray
        Full energy-resolved prompt sinogram.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo ground-truth components.
    energies : array-like, optional
        Energy-bin centres. Defaults to 410, 430, ..., 590 keV.
    dE : float, optional
        Energy-bin width in keV. Default is 20.
    
    Returns
    -------
    numpy.ndarray
        Regional NEGML coefficient estimates with shape ``(4,) + sino_shape``.
    
    Notes
    -----
    The current implementation explicitly partitions a 32-by-55
    view--tangential plane and averages the prompt data over the axial axis
    before regional fitting.
    """
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
    """Fit one selected spatial mask using a chosen basis construction.
    
    This diagnostic workflow derives the requested true or predicted PDFs
    within ``mask``, constructs the selected shared, separate, or direct 2D
    basis, and fits a four-parameter NEGML model to the region's joint-energy
    histogram.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        Spatial mask over the view--tangential plane.
    Func_derive_basis : callable
        Basis-construction function.
    PDF_nonscatter_true_or_predicted : {"true", "predicted"}
        Source of the non-scatter PDF.
    PDF_nonscatter : array-like
        Predicted non-scatter PDF used when requested.
    PDF_scatter_true_or_predicted : {"true", "predicted"}
        Source of the scatter PDF.
    sino_shape : tuple
        Spatial sinogram shape.
    sinogram_with_energy_bin_info : numpy.ndarray
        Full energy-resolved prompt data.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo scatter-history components.
    energies : array-like, optional
        Energy-bin centres.
    dE : float, optional
        Energy-bin width in keV.
    
    Returns
    -------
    numpy.ndarray
        Four fitted NEGML coefficients for the selected region.
    
    Notes
    -----
    This function is intended for representative-pixel diagnostics rather than
    for constructing a full spatial coefficient sinogram.
    """
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