"""
Energy PDF estimation utilities.

Parts of the non-scatter and scatter PDF estimation implemented in this
module are adapted from the previous work of Yifan Ding:
[GitHub repository URL].

The original notebook-based implementation has been reorganised and
extended for the data format and regional PDF analysis used in this project.
"""

# import sirf.STIR as PET
import numpy as np
import math

def derive_true_components(blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1, energies):
    blue0_pink0_1D = np.sum(blue0_pink0, axis = 1)
    blue0_pink1_1D = np.sum(blue0_pink1, axis = 1)
    blue1_pink0_1D = np.sum(blue1_pink0, axis = 1)
    blue1_pink1_1D = np.sum(blue1_pink1, axis = 1)
    non_scatter_component = (blue0_pink0_1D + blue0_pink1_1D) / np.maximum(np.trapz((blue0_pink0_1D + blue0_pink1_1D), energies), 1e-10)
    scatter_component = (blue1_pink0_1D + blue1_pink1_1D)/ np.maximum(np.trapz((blue1_pink0_1D + blue1_pink1_1D), energies), 1e-10)
    return non_scatter_component, scatter_component



def estimate_nonscatter_PDF(resolution_percentage, energies):
    # Estimate non-scatter PDF
    FWHM = 511 * 0.01 * resolution_percentage #det.rec - energy_resolution_percentage = ?
    sigma_511 = FWHM / math.sqrt(8 * math.log(2))

    PDF_nonscatter = []
    for i in (energies):
        PDF_nonscatter.append((1 / (math.sqrt(2 * math.pi) * sigma_511)) * math.exp((- ((i - 511) ** 2) / (2 * sigma_511 ** 2)))) #i: E
    #U(E) every 20keV at the midpoint
    PDF_nonscatter = PDF_nonscatter/ np.trapz(PDF_nonscatter, energies)

    return PDF_nonscatter


def estimate_scatter_PDF(total_1D_histogram, PDF_nonscatter, energies): 
    """
    Estimate the scatter energy PDF.

    Adapted from the PDF-estimation procedure implemented by
    Ian [Surname] in [repository/notebook reference], with modifications
    for the analysis workflow used in this project.
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


def predict_pdf_scatter_within_mask(PDF_nonscatter, total_histogram_inside_mask, energies):
    total_histogram_inside_mask_1d = np.sum(total_histogram_inside_mask, axis=1)

    PDF_scatter_inside_mask = estimate_scatter_PDF(total_histogram_inside_mask_1d, PDF_nonscatter, energies)
    return PDF_scatter_inside_mask
