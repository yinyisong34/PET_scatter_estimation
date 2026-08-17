import numpy as np

def convert_PDF_to_bin_probability(PDF, dE):
    PDF_bin_prob = (np.asarray(PDF) * dE) / np.maximum((np.sum((np.asarray(PDF) * dE))), 1e-8)
    return PDF_bin_prob

def derive_shared_1d_factorised_basis(U_E, S_E, dE):
    U_bin = convert_PDF_to_bin_probability(U_E, dE)
    S_bin = convert_PDF_to_bin_probability(S_E, dE)

    C0 = np.outer(U_bin, U_bin) # U(E1)U(E2)
    C1 = np.outer(U_bin, S_bin) # U(E1)S(E2)
    C2 = np.outer(S_bin, U_bin) # S(E1)U(E2)
    C3 = np.outer(S_bin, S_bin) # S(E1)S(E2)

    basis = np.stack([C0, C1, C2, C3], axis=0)
    return basis

def derive_separate_1d_factorised_basis(U1_E, S1_E, U2_E, S2_E, dE):
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