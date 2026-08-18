# SPDX-License-Identifier: Apache-2.0
"""Utilities for constructing energy-domain basis functions for EBSE.

This module converts sampled one-dimensional energy PDFs into discrete bin
probabilities and constructs the four basis components used by the
four-parameter NEGML model. It supports shared 1D factorised, separate 1D
factorised, and direct simulation-derived 2D basis constructions."""

import numpy as np

def convert_PDF_to_bin_probability(PDF, dE):
    """Convert sampled PDF values to discrete energy-bin probabilities.
    
    Parameters
    ----------
    PDF : array-like
        Sampled probability-density values at the energy-bin locations.
    dE : float
        Energy-bin width in keV.
    
    Returns
    -------
    numpy.ndarray
        Normalised probability mass in each energy bin. The returned values sum
        to approximately one.
    
    Notes
    -----
    The conversion multiplies each sampled density by ``dE`` and renormalises
    the result. A small numerical floor is used in the denominator to avoid
    division by zero.
    """
    PDF_bin_prob = (np.asarray(PDF) * dE) / np.maximum((np.sum((np.asarray(PDF) * dE))), 1e-8)
    return PDF_bin_prob

def derive_shared_1d_factorised_basis(U_E, S_E, dE):
    """Construct a shared 1D factorised four-component energy basis.
    
    The same non-scatter PDF ``U_E`` and scatter PDF ``S_E`` are used for both
    photon-energy dimensions. The four joint basis components are formed as
    outer products: UU, US, SU, and SS.
    
    Parameters
    ----------
    U_E : array-like
        One-dimensional non-scatter energy PDF, U(E).
    S_E : array-like
        One-dimensional scatter energy PDF, S(E).
    dE : float
        Energy-bin width in keV.
    
    Returns
    -------
    numpy.ndarray
        Array of shape ``(4, n_energy, n_energy)`` containing the basis
        components ``[UU, US, SU, SS]``.
    """
    U_bin = convert_PDF_to_bin_probability(U_E, dE)
    S_bin = convert_PDF_to_bin_probability(S_E, dE)

    C0 = np.outer(U_bin, U_bin) # U(E1)U(E2)
    C1 = np.outer(U_bin, S_bin) # U(E1)S(E2)
    C2 = np.outer(S_bin, U_bin) # S(E1)U(E2)
    C3 = np.outer(S_bin, S_bin) # S(E1)S(E2)

    basis = np.stack([C0, C1, C2, C3], axis=0)
    return basis

def derive_separate_1d_factorised_basis(U1_E, S1_E, U2_E, S2_E, dE):
    """Construct a factorised basis using separate PDFs for the two photons.
    
    Separate non-scatter and scatter PDFs are retained for the first and second
    photon-energy dimensions. The four basis components are U1U2, U1S2,
    S1U2, and S1S2.
    
    Parameters
    ----------
    U1_E, S1_E : array-like
        Non-scatter and scatter PDFs for the first photon-energy dimension.
    U2_E, S2_E : array-like
        Non-scatter and scatter PDFs for the second photon-energy dimension.
    dE : float
        Energy-bin width in keV.
    
    Returns
    -------
    numpy.ndarray
        Array of shape ``(4, n_energy, n_energy)`` containing
        ``[U1U2, U1S2, S1U2, S1S2]``.
    """
    U1_bin = convert_PDF_to_bin_probability(U1_E, dE)
    S1_bin = convert_PDF_to_bin_probability(S1_E, dE)
    U2_bin = convert_PDF_to_bin_probability(U2_E, dE)
    S2_bin = convert_PDF_to_bin_probability(S2_E, dE)

    C0 = np.outer(U1_bin, U2_bin) # U1(E1)U2(E2)
    C1 = np.outer(U1_bin, S2_bin) # U1(E1)S2(E2)
    C2 = np.outer(S1_bin, U2_bin) # S1(E1)U2(E2)
    C3 = np.outer(S1_bin, S2_bin) # S1(E1)S2(E2)

    basis = np.stack([C0, C1, C2, C3], axis=0)
    return basis

def derive_ground_truth_2d_basis(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Construct direct 2D ground-truth basis functions within a spatial region.
    
    The energy-resolved Monte Carlo components C0--C3 are restricted to the
    selected view--tangential region and summed over the remaining spatial
    dimensions. Each resulting joint energy histogram is independently
    normalised and used directly as one basis component.
    
    Parameters
    ----------
    mask : numpy.ndarray of bool
        Two-dimensional mask over the final view--tangential spatial axes.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo sinograms. Their first two dimensions are
        the E1 and E2 energy bins; the final two dimensions correspond to the
        spatial plane selected by ``mask``.
    
    Returns
    -------
    numpy.ndarray
        Array of shape ``(4, n_energy, n_energy)`` containing the independently
        normalised C0, C1, C2, and C3 joint-energy basis functions.
    
    Notes
    -----
    Unlike the 1D basis constructors, this function does not factorise the
    joint energy distribution into products of marginal PDFs.
    """
    C0_histogram = np.sum(c0_sinogram_with_energy_info * mask[None, None, None, None, :, :], axis=(2,3,4,5))
    C1_histogram = np.sum(c1_sinogram_with_energy_info * mask[None, None, None, None, :, :], axis=(2,3,4,5))
    C2_histogram = np.sum(c2_sinogram_with_energy_info * mask[None, None, None, None, :, :], axis=(2,3,4,5))
    C3_histogram = np.sum(c3_sinogram_with_energy_info * mask[None, None, None, None, :, :], axis=(2,3,4,5))

    C0 = C0_histogram / np.maximum(np.sum(C0_histogram), 1e-8)
    C1 = C1_histogram / np.maximum(np.sum(C1_histogram), 1e-8)
    C2 = C2_histogram / np.maximum(np.sum(C2_histogram), 1e-8)
    C3 = C3_histogram / np.maximum(np.sum(C3_histogram), 1e-8)

    basis = np.stack([C0, C1, C2, C3], axis=0)
    return basis