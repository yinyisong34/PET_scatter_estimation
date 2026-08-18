import numpy as np

def creat_mask_at(i,j):
    at_i_j = np.zeros((32,55))  # shape (32,55)
    at_i_j[i,j] = 1
    mask = at_i_j > np.zeros((32,55))     # shape (32,55)
    return mask

def nonscatter_and_scatter_histogram_inside_mask(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
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
    # (10, 10, 1, 12, 32, 55)
    mask_expanded = mask[None, None, None, None, :, :] 
    sinogram_inside_mask = sinogram_with_energy_bin_info * mask_expanded
    return sinogram_inside_mask

def c0123_sinogram_with_energy_info_in(mask, c0_sinogram_with_energy_info, c1_sinogram_with_energy_info, c2_sinogram_with_energy_info, c3_sinogram_with_energy_info):
    # (10, 10, 1, 12, 32, 55)
    c0_sinogram_with_energy_info_in = mask_the_sinogram_with_energy_bin_info(mask, c0_sinogram_with_energy_info)
    c1_sinogram_with_energy_info_in= mask_the_sinogram_with_energy_bin_info(mask, c1_sinogram_with_energy_info)
    c2_sinogram_with_energy_info_in= mask_the_sinogram_with_energy_bin_info(mask, c2_sinogram_with_energy_info)
    c3_sinogram_with_energy_info_in = mask_the_sinogram_with_energy_bin_info(mask, c3_sinogram_with_energy_info)
    return c0_sinogram_with_energy_info_in, c1_sinogram_with_energy_info_in, c2_sinogram_with_energy_info_in, c3_sinogram_with_energy_info_in

