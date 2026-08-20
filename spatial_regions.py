# SPDX-License-Identifier: Apache-2.0
"""Spatial masking and regional energy-distribution utilities.

This module creates view--tangential masks, extracts energy-resolved data
within selected regions, and derives regional ground-truth energy
histograms and one-dimensional PDFs from the Monte Carlo C0--C3
components."""

import numpy as np

def create_mask_at(i,j):
    """Create a one-pixel boolean mask on the 32-by-55 spatial plane.
    
    Parameters
    ----------
    i : int
        Row/view index.
    j : int
        Column/tangential index.
    
    Returns
    -------
    numpy.ndarray of bool
        Boolean mask of shape ``(32, 55)`` with only ``(i, j)`` set to True.
    """
    at_i_j = np.zeros((32,55))  # shape (32,55)
    at_i_j[i,j] = 1
    mask = at_i_j > np.zeros((32,55))     # shape (32,55)
    return mask

def nonscatter_and_scatter_histogram_inside_mask(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Aggregate joint-energy histograms inside a mask by first-photon scatter state.
    
    For each E1/E2 bin, the C0 and C1 components are combined as cases in which
    the first (blue) photon is unscattered, while C2 and C3 are combined as
    cases in which it is scattered. The first segment is used and the axial
    dimension is averaged before the selected spatial pixels are summed.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        View--tangential spatial mask.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo components.
    
    Returns
    -------
    nonscatter_in : numpy.ndarray
        Regional joint-energy histogram for an unscattered first photon.
    scatter_in : numpy.ndarray
        Regional joint-energy histogram for a scattered first photon.
    """
    scatter_in = np.zeros((10,10))
    nonscatter_in = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            ns = (
                c0_sinogram_with_energy_info[i, j] 
                + c1_sinogram_with_energy_info[i, j] 
            )

            sc = (
                c2_sinogram_with_energy_info[i, j] 
                + c3_sinogram_with_energy_info[i, j] 
            )

            ns_avg = np.mean(ns[0,:,:,:], axis=0)    # shape (32,55)
            sc_avg = np.mean(sc[0,:,:,:], axis=0)

            nonscatter_in[i,j] = np.sum(ns_avg[mask])
            scatter_in[i,j] = np.sum(sc_avg[mask])

    return nonscatter_in, scatter_in


def total_histogram_inside_mask(mask, Y):   
    """Extract the total joint-energy histogram from a selected spatial region.
    
    For each E1/E2 bin, all axes except the final two spatial axes are summed.
    The remaining view--tangential plane is then restricted by ``mask`` and
    summed to one value.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        Mask over the final two spatial axes.
    Y : numpy.ndarray
        Energy-resolved total prompt data. The first two axes must be E1 and E2;
        the final two axes must match ``mask``.
    
    Returns
    -------
    numpy.ndarray
        Regional joint-energy histogram with shape ``(n_energy, n_energy)``.
    """
    total_histogram_inside_mask = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            sinogram_i_j = Y[i,j] # Y shape (1, 144, 32, 55) or Y_avg shape (1, 32, 55)

            # Sum over every axis except the final two spatial axes.
            # (1, 32, 55)      -> sum axis (0,)   -> (32, 55)
            # (1, 144, 32, 55) -> sum axes (0, 1) -> (32, 55)
            axes_to_sum = tuple(range(sinogram_i_j.ndim - 2))
            sinogram_i_j_2d = np.sum(
                sinogram_i_j,
                axis=axes_to_sum
            )

            total_histogram_inside_mask[i,j] = np.sum(sinogram_i_j_2d[mask])

    return total_histogram_inside_mask




def true_pdf_inside_mask(mask, energies, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Derive ground-truth U1, S1, U2, and S2 PDFs inside a spatial mask.
    
    The C0--C3 simulation components are grouped by the scatter state of the
    first (blue) and second (pink) photons. The first segment is used, the axial
    dimension is averaged, the selected spatial pixels are summed, and the
    resulting joint histograms are marginalised and normalised.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        View--tangential spatial mask.
    energies : array-like
        Energy-bin centres used for numerical normalisation.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo ground-truth components.
    
    Returns
    -------
    nonscatter1_in_pdf : numpy.ndarray
        Ground-truth non-scatter PDF for the first photon-energy dimension.
    scatter1_in_pdf : numpy.ndarray
        Ground-truth scatter PDF for the first photon-energy dimension.
    nonscatter2_in_pdf : numpy.ndarray
        Ground-truth non-scatter PDF for the second photon-energy dimension.
    scatter2_in_pdf : numpy.ndarray
        Ground-truth scatter PDF for the second photon-energy dimension.
    """
    scatter_blue_in = np.zeros((10,10))
    nonscatter_blue_in = np.zeros((10,10))
    scatter_pink_in = np.zeros((10,10))
    nonscatter_pink_in = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            nonscatter_blue = c0_sinogram_with_energy_info[i, j] + c1_sinogram_with_energy_info[i, j] 
            scatter_blue = c2_sinogram_with_energy_info[i, j] + c3_sinogram_with_energy_info[i, j] 
            nonscatter_pink = c0_sinogram_with_energy_info[i, j] + c2_sinogram_with_energy_info[i, j] 
            scatter_pink = c1_sinogram_with_energy_info[i, j] + c3_sinogram_with_energy_info[i, j] 
            
            ns_avg_blue = np.mean(nonscatter_blue[0,:,:,:], axis=0)    # shape (32,55)
            sc_avg_blue = np.mean(scatter_blue[0,:,:,:], axis=0)
            ns_avg_pink = np.mean(nonscatter_pink[0,:,:,:], axis=0)   
            sc_avg_pink = np.mean(scatter_pink[0,:,:,:], axis=0)

            nonscatter_blue_in[i,j] = np.sum(ns_avg_blue[mask])
            scatter_blue_in[i,j] = np.sum(sc_avg_blue[mask])

            nonscatter_pink_in[i,j] = np.sum(ns_avg_pink[mask])
            scatter_pink_in[i,j] = np.sum(sc_avg_pink[mask])

    # 1D histogram
    nonscatter1_in_1d = np.sum(nonscatter_blue_in, axis=1)
    scatter1_in_1d = np.sum(scatter_blue_in, axis=1)

    nonscatter2_in_1d = np.sum(nonscatter_pink_in, axis=0)
    scatter2_in_1d = np.sum(scatter_pink_in, axis=0)


    # true pdf
    nonscatter1_in_pdf = nonscatter1_in_1d / np.maximum(np.trapz(nonscatter1_in_1d, energies), 1e-8)
    scatter1_in_pdf = scatter1_in_1d / np.maximum(np.trapz(scatter1_in_1d, energies), 1e-8)

    nonscatter2_in_pdf = nonscatter2_in_1d / np.maximum(np.trapz(nonscatter2_in_1d, energies), 1e-8)
    scatter2_in_pdf = scatter2_in_1d / np.maximum(np.trapz(scatter2_in_1d, energies), 1e-8)

    return nonscatter1_in_pdf, scatter1_in_pdf, nonscatter2_in_pdf, scatter2_in_pdf


def mask_the_sinogram_with_energy_bin_info(mask, sinogram_with_energy_bin_info):
    """Apply a 2D spatial mask to an energy-resolved sinogram.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        Mask over the view--tangential plane.
    sinogram_with_energy_bin_info : numpy.ndarray
        Energy-resolved sinogram, typically shaped
        ``(E1, E2, segment, axial, view, tangential)``.
    
    Returns
    -------
    numpy.ndarray
        Array with the same shape as the input sinogram, with values outside the
        selected spatial mask set to zero.
    """
    # sinogram_with_energy_bin_info shape (10, 10, 1, 12, 32, 55)
    mask_expanded = mask[None, None, None, None, :, :] 
    sinogram_inside_mask = sinogram_with_energy_bin_info * mask_expanded
    return sinogram_inside_mask

def c0123_sinogram_with_energy_info_in(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Apply the same spatial mask to all four Monte Carlo energy components.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        View--tangential spatial mask.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved C0--C3 sinograms.
    
    Returns
    -------
    c0_sinogram_with_energy_info_in, c1_sinogram_with_energy_info_in, c2_sinogram_with_energy_info_in, c3_sinogram_with_energy_info_in : numpy.ndarray
        Masked copies of the four input arrays.
    """
    # sinogram_with_energy_bin_info shape (10, 10, 1, 12, 32, 55)
    c0_sinogram_with_energy_info_in = mask_the_sinogram_with_energy_bin_info(mask, c0_sinogram_with_energy_info)
    c1_sinogram_with_energy_info_in= mask_the_sinogram_with_energy_bin_info(mask, c1_sinogram_with_energy_info)
    c2_sinogram_with_energy_info_in= mask_the_sinogram_with_energy_bin_info(mask, c2_sinogram_with_energy_info)
    c3_sinogram_with_energy_info_in = mask_the_sinogram_with_energy_bin_info(mask, c3_sinogram_with_energy_info)
    return c0_sinogram_with_energy_info_in, c1_sinogram_with_energy_info_in, c2_sinogram_with_energy_info_in, c3_sinogram_with_energy_info_in

