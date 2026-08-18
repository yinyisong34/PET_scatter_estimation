"""Energy PDF estimation utilities for the EBSE analysis.

This module derives simulation-based ground-truth one-dimensional energy
PDFs and estimates the predicted non-scatter and scatter PDFs used to build
the factorised energy basis functions.

Parts of the non-scatter and scatter PDF estimation implemented in this
module are adapted from the previous work of Yifan Ding:
[GitHub repository URL].

The original notebook-based implementation has been reorganised and
extended for the data format and regional PDF analysis used in this project."""

# import sirf.STIR as PET
import numpy as np
import math

def derive_true_components(blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1, energies):
    """Derive global ground-truth 1D PDFs for the first photon dimension.
    
    The two Monte Carlo categories in which the first (blue) photon is
    unscattered are combined to form the non-scatter distribution, while the
    two categories in which it is scattered are combined to form the scatter
    distribution. The second photon-energy dimension is marginalised out.
    
    Parameters
    ----------
    blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1 : numpy.ndarray
        Global joint-energy histograms for the four photon scatter-history
        combinations.
    energies : array-like
        Energy-bin centre values used for numerical normalisation.
    
    Returns
    -------
    non_scatter_component : numpy.ndarray
        Normalised ground-truth non-scatter PDF for the first photon dimension.
    scatter_component : numpy.ndarray
        Normalised ground-truth scatter PDF for the first photon dimension.
    """
    blue0_pink0_1D = np.sum(blue0_pink0, axis = 1)
    blue0_pink1_1D = np.sum(blue0_pink1, axis = 1)
    blue1_pink0_1D = np.sum(blue1_pink0, axis = 1)
    blue1_pink1_1D = np.sum(blue1_pink1, axis = 1)
    non_scatter_component = (blue0_pink0_1D + blue0_pink1_1D) / np.maximum(np.trapz((blue0_pink0_1D + blue0_pink1_1D), energies), 1e-10)
    scatter_component = (blue1_pink0_1D + blue1_pink1_1D)/ np.maximum(np.trapz((blue1_pink0_1D + blue1_pink1_1D), energies), 1e-10)
    return non_scatter_component, scatter_component



def estimate_nonscatter_PDF(resolution_percentage, energies):
    # Estimate non-scatter PDF
    """Estimate the non-scatter energy PDF from detector energy resolution.
    
    The unscattered photopeak is modelled as a Gaussian centred at 511 keV.
    The supplied detector resolution is interpreted as a percentage FWHM at
    511 keV and converted to the corresponding Gaussian standard deviation.
    
    Parameters
    ----------
    resolution_percentage : float
        Detector energy resolution in percent FWHM at 511 keV.
    energies : array-like
        Energy-bin centre values at which the Gaussian is evaluated.
    
    Returns
    -------
    numpy.ndarray
        Numerically normalised non-scatter PDF evaluated at ``energies``.
    """
    FWHM = 511 * 0.01 * resolution_percentage #det.rec - energy_resolution_percentage = ?
    sigma_511 = FWHM / math.sqrt(8 * math.log(2))

    PDF_nonscatter = []
    for i in (energies):
        PDF_nonscatter.append((1 / (math.sqrt(2 * math.pi) * sigma_511)) * math.exp((- ((i - 511) ** 2) / (2 * sigma_511 ** 2)))) #i: E
    #U(E) every 20keV at the midpoint
    PDF_nonscatter = PDF_nonscatter/ np.trapz(PDF_nonscatter, energies)

    return PDF_nonscatter


def estimate_scatter_PDF(total_1D_histogram, PDF_nonscatter, energies): 
    """Estimate the scatter energy PDF from a marginal total-energy spectrum.
    
    The supplied non-scatter PDF is scaled using the four highest energy bins
    (indices 6--9), subtracted from the total marginal spectrum, clipped at
    zero, and numerically normalised to obtain the residual scatter PDF.
    
    Parameters
    ----------
    total_1D_histogram : array-like
        One-dimensional total prompt-energy histogram.
    PDF_nonscatter : array-like
        Predicted or ground-truth non-scatter PDF sampled at ``energies``.
    energies : array-like
        Energy-bin centre values used for numerical normalisation.
    
    Returns
    -------
    numpy.ndarray
        Estimated, non-negative, normalised scatter energy PDF.
    
    Notes
    -----
    Adapted from the PDF-estimation procedure implemented by Yifan Ding, with modifications for the analysis
    workflow used in this project. The high-energy scaling region is fixed by
    the current 10-bin implementation.
    """
    # Find the total contribution of unscattered photons and compare it with (c_0 + c_1)
    total_1D_histogram_window = 0
    for i in range(6,10):
        total_1D_histogram_window += total_1D_histogram[i]

    unscattered_total_con = total_1D_histogram_window / np.sum(PDF_nonscatter[6:])

    PDF_scatter_scaled = total_1D_histogram - np.array(PDF_nonscatter) * unscattered_total_con
    PDF_scatter_scaled = np.maximum(PDF_scatter_scaled, 0)
    area = np.trapz(PDF_scatter_scaled, energies)
    PDF_scatter = PDF_scatter_scaled / np.maximum(area, 1e-8)
    return PDF_scatter