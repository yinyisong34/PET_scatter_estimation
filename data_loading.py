"""Loading and aggregation utilities for energy-resolved SimSET/STIR data.

The functions in this module read the converted energy-resolved projection
files, assemble the four Monte Carlo scatter-history components C0--C3,
and derive global energy histograms or spatial sinograms used by the EBSE
analysis."""

import sirf.STIR as PET
import numpy as np
from pathlib import Path

def derive_sinogram_with_energy_bin_info_full(folder_path):
    """Load all energy-resolved C0--C3 projection files from a simulation folder.
    
    The expected filenames encode the energy bins and scatter histories of the
    two detected photons. Ten E1 bins and ten E2 bins are loaded for each of the
    four Monte Carlo categories.
    
    Parameters
    ----------
    folder_path : str or pathlib.Path
        Directory containing the converted STIR ``.hs`` acquisition files.
    
    Returns
    -------
    sino_shape : tuple
        Shape of one spatial sinogram, typically ordered as
        ``(segment, axial, view, tangential)``.
    Y_full : numpy.ndarray
        Total energy-resolved prompt data obtained as C0 + C1 + C2 + C3.
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved sinograms for the four scatter-history categories.
        Each array has shape ``(10, 10) + sino_shape``.
    
    Raises
    ------
    FileNotFoundError
        If the sample file or any expected energy/scatter-category file is
        missing.
    """
    folder_path = Path(folder_path)

    # Read one file first to determine the sinogram shape
    sample_file = folder_path / "blue0_en5_pink0_en5.hs"

    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    sinogram = PET.AcquisitionData(str(sample_file)).as_array()
    sino_shape = sinogram.shape

    c0_sinogram_with_energy_info = np.zeros((10, 10) + sino_shape)
    c1_sinogram_with_energy_info = np.zeros((10, 10) + sino_shape)
    c2_sinogram_with_energy_info = np.zeros((10, 10) + sino_shape)
    c3_sinogram_with_energy_info = np.zeros((10, 10) + sino_shape)

    for i in range(10):
        for j in range(10):
            filenames = {
                "c0": folder_path / f"blue0_en{i}_pink0_en{j}.hs",
                "c1": folder_path / f"blue0_en{i}_pink1_en{j}.hs",
                "c2": folder_path / f"blue1_en{i}_pink0_en{j}.hs",
                "c3": folder_path / f"blue1_en{i}_pink1_en{j}.hs",
            }

            for name, filepath in filenames.items():
                if not filepath.exists():
                    raise FileNotFoundError(
                        f"Missing {name} file for i={i}, j={j}: {filepath}"
                    )

            c0_sinogram_with_energy_info[i, j] = PET.AcquisitionData(str(filenames["c0"])).as_array()
            c1_sinogram_with_energy_info[i, j] = PET.AcquisitionData(str(filenames["c1"])).as_array()
            c2_sinogram_with_energy_info[i, j] = PET.AcquisitionData(str(filenames["c2"])).as_array()
            c3_sinogram_with_energy_info[i, j] = PET.AcquisitionData(str(filenames["c3"])).as_array()

    Y_full = (c0_sinogram_with_energy_info
            + c1_sinogram_with_energy_info
            + c2_sinogram_with_energy_info
            + c3_sinogram_with_energy_info)

    return (
        sino_shape,
        Y_full,
        c0_sinogram_with_energy_info,
        c1_sinogram_with_energy_info,
        c2_sinogram_with_energy_info,
        c3_sinogram_with_energy_info,
    )


def derive_sinogram(c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Collapse the energy dimensions to obtain spatial ground-truth sinograms.
    
    Parameters
    ----------
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved C0--C3 arrays with E1 and E2 as the first two axes.
    
    Returns
    -------
    c0_sinogram, c1_sinogram, c2_sinogram, c3_sinogram : numpy.ndarray
        Spatial sinograms for each Monte Carlo scatter-history category.
    nonscatter_sinogram : numpy.ndarray
        Ground-truth non-scatter sinogram, equal to C0.
    scatter_sinogram : numpy.ndarray
        Ground-truth total scatter sinogram, equal to C1 + C2 + C3.
    total_sinogram : numpy.ndarray
        Sum of the non-scatter and scatter sinograms.
    """
    c0_sinogram = np.sum(c0_sinogram_with_energy_info, axis=(0,1))
    c1_sinogram = np.sum(c1_sinogram_with_energy_info, axis=(0,1))
    c2_sinogram = np.sum(c2_sinogram_with_energy_info, axis=(0,1))
    c3_sinogram = np.sum(c3_sinogram_with_energy_info, axis=(0,1))

    nonscatter_sinogram = c0_sinogram
    scatter_sinogram = c1_sinogram + c2_sinogram + c3_sinogram
    total_sinogram = nonscatter_sinogram + scatter_sinogram
    return (
        c0_sinogram, 
        c1_sinogram, 
        c2_sinogram, 
        c3_sinogram, 
        nonscatter_sinogram, 
        scatter_sinogram, 
        total_sinogram,
        )


def derive_blue_pink(c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    """Collapse all spatial dimensions to obtain global joint-energy histograms.
    
    Parameters
    ----------
    c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info : numpy.ndarray
        Energy-resolved Monte Carlo sinograms.
    
    Returns
    -------
    blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1 : numpy.ndarray
        Global ``(E1, E2)`` histograms for the four combinations of photon
        scatter history. With the current 10-bin data these arrays have shape
        ``(10, 10)``.
    """
    blue0_pink0 = np.sum(c0_sinogram_with_energy_info, axis=(2,3,4,5))
    blue0_pink1 = np.sum(c1_sinogram_with_energy_info, axis=(2,3,4,5))
    blue1_pink0 = np.sum(c2_sinogram_with_energy_info, axis=(2,3,4,5))
    blue1_pink1 = np.sum(c3_sinogram_with_energy_info, axis=(2,3,4,5))
    return blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1


