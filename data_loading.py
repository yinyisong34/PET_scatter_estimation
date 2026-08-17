import sirf.STIR as PET
import numpy as np
from pathlib import Path

def derive_sinogram_with_energy_bin_info_full(folder_path):
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
    blue0_pink0 = np.sum(c0_sinogram_with_energy_info, axis=(2,3,4,5))
    blue0_pink1 = np.sum(c1_sinogram_with_energy_info, axis=(2,3,4,5))
    blue1_pink0 = np.sum(c2_sinogram_with_energy_info, axis=(2,3,4,5))
    blue1_pink1 = np.sum(c3_sinogram_with_energy_info, axis=(2,3,4,5))
    return blue0_pink0, blue0_pink1, blue1_pink0, blue1_pink1