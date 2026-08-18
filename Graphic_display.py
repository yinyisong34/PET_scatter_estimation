import numpy as np
import matplotlib.pyplot as plt

def sigma_0123(sigma_negml):
    sigma0 = sigma_negml[0]
    sigma1 = sigma_negml[1]
    sigma2 = sigma_negml[2]
    sigma3 = sigma_negml[3]

    sigma_scatter = sigma1 + sigma2 + sigma3

    return sigma0, sigma1, sigma2, sigma3, sigma_scatter


def get_negml_labels(
    basis_function,
    U_true_or_predict,
    S_true_or_predict,
    pixel_width,
    pixel_length,
    sino_shape
):
    """
    Generate labels automatically from the actual NEGML configuration.
    """

    # ---------------------------------------------------------
    # U and S source labels
    # ---------------------------------------------------------

    pdf_source_labels = {
        "true": "Ground-truth",
        "predicted": "Predicted"
    }

    U_label = pdf_source_labels.get(
        U_true_or_predict,
        U_true_or_predict
    )

    S_label = pdf_source_labels.get(
        S_true_or_predict,
        S_true_or_predict
    )


    # ---------------------------------------------------------
    # Basis construction label
    # ---------------------------------------------------------

    basis_name = basis_function.__name__

    basis_labels = {
        "derive_shared_1d_factorised_basis":
            "Shared 1D factorised basis",

        "derive_separate_1d_factorised_basis":
            "Separate 1D factorised basis",

        "derive_ground_truth_2d_basis":
            "Ground-truth 2D basis"
    }

    basis_label = basis_labels.get(
        basis_name,
        basis_name
    )


    # ---------------------------------------------------------
    # Spatial model label
    # sino_shape = (segment, axial, view, tangential)
    # ---------------------------------------------------------

    n_view = sino_shape[-2]
    n_tangential = sino_shape[-1]

    if (
        pixel_width == n_view
        and pixel_length == n_tangential
    ):
        spatial_label = "Global basis model"
    else:
        spatial_label = (
            f"Multi-region basis model "
            f"({pixel_width} × {pixel_length} pixels per region)"
        )


    return (
        U_label,
        S_label,
        basis_label,
        spatial_label
    )


def plot_negml_results(
    scatter_sinogram,
    nonscatter_sinogram,
    sigma_scatter_negml,
    sigma0_negml,
    U_label,
    S_label,
    basis_label,
    spatial_label,
    view_idx=16
):

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))


    # 1. Scatter at selected view
    axes[0, 0].plot(
        np.mean(
            scatter_sinogram[0, :, view_idx, :],
            axis=0
        ),
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].plot(
        np.mean(
            sigma_scatter_negml[0, :, view_idx, :],
            axis=0
        ),
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].set_title(
        f"Scatter: view {view_idx}"
    )


    # 2. Scatter averaged over axial and view
    axes[0, 1].plot(
        np.mean(
            scatter_sinogram[0],
            axis=(0, 1)
        ),
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].plot(
        np.mean(
            sigma_scatter_negml[0],
            axis=(0, 1)
        ),
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].set_title(
        "Scatter: averaged over axial and view"
    )


    # 3. Non-scatter at selected view
    axes[1, 0].plot(
        np.mean(
            nonscatter_sinogram[0, :, view_idx, :],
            axis=0
        ),
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].plot(
        np.mean(
            sigma0_negml[0, :, view_idx, :],
            axis=0
        ),
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].set_title(
        f"Non-scatter: view {view_idx}"
    )


    # 4. Non-scatter averaged over axial and view
    axes[1, 1].plot(
        np.mean(
            nonscatter_sinogram[0],
            axis=(0, 1)
        ),
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].plot(
        np.mean(
            sigma0_negml[0],
            axis=(0, 1)
        ),
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].set_title(
        "Non-scatter: averaged over axial and view"
    )


    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    for ax in axes.flat:
        ax.legend(frameon=False)
        ax.tick_params(
            axis="both",
            labelsize=11
        )


    fig.suptitle(
        "NEGML results\n"
        f"{spatial_label}\n"
        f"{basis_label}\n"
        f"U(E): {U_label}   |   S(E): {S_label}",
        fontsize=15
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)




import numpy as np
import matplotlib.pyplot as plt

def plot_basis(pixel, basis_type, basis_idx, resolution=10, basis_data=None):
    """
    Plot one basis image with an automatically generated title.

    Parameters
    ----------
    pixel : tuple
        Pixel location, e.g. (16, 10)
    basis_type : str
        One of: "shared", "separate", "GT_2D"
    basis_idx : int
        Basis index: 0, 1, 2, or 3
    resolution : int
        Used only if basis_data is None, to construct variable name automatically
    basis_data : array-like, optional
        If provided, use this directly.
        If None, the function will try to find a global variable named:
        basis_resolution_{resolution}_pixel_{i}_{j}_{basis_type}
    """

    i, j = pixel

    # Allow a few aliases
    basis_type_alias = {
        "shared": "shared",
        "separate": "separate",
        "GT_2D": "GT_2D",
        "ground_truth": "GT_2D",
        "ground-truth": "GT_2D"
    }

    if basis_type not in basis_type_alias:
        raise ValueError("basis_type must be one of: 'shared', 'separate', 'GT_2D'")

    basis_type = basis_type_alias[basis_type]

    # Title prefix and formula mapping
    title_prefix = {
        "shared": "Shared factorised basis",
        "separate": "Separate factorised basis",
        "GT_2D": "Ground-truth 2D basis"
    }

    title_formula = {
        "shared": [
            r"U_1(E_1)U_1(E_2)",
            r"U_1(E_1)S_1(E_2)",
            r"S_1(E_1)U_1(E_2)",
            r"S_1(E_1)S_1(E_2)"
        ],
        "separate": [
            r"U_1(E_1)U_2(E_2)",
            r"U_1(E_1)S_2(E_2)",
            r"S_1(E_1)U_2(E_2)",
            r"S_1(E_1)S_2(E_2)"
        ],
        "GT_2D": [
            r"C_0(E_1,E_2)",
            r"C_1(E_1,E_2)",
            r"C_2(E_1,E_2)",
            r"C_3(E_1,E_2)"
        ]
    }

    if basis_idx not in [0, 1, 2, 3]:
        raise ValueError("basis_idx must be 0, 1, 2, or 3")

    if basis_data is None:
        import inspect

        var_name = f"basis_resolution_{resolution}_pixel_{i}_{j}_{basis_type}"

        # Get the namespace of the notebook/script that called this function
        caller_frame = inspect.currentframe().f_back
        caller_namespace = caller_frame.f_globals

        if var_name not in caller_namespace:
            raise ValueError(
                f"Cannot find variable '{var_name}' in the caller namespace."
            )

        basis_data = caller_namespace[var_name]

    plt.figure(figsize=(6, 5))
    plt.imshow(basis_data[basis_idx])
    plt.colorbar()
    plt.title(
        rf"{title_prefix[basis_type]} ${title_formula[basis_type][basis_idx]}$ at pixel ({i}, {j})",
        fontsize=15,
        pad=20
    )
    plt.xticks(np.arange(0, basis_data[basis_idx].shape[1], 1), fontsize=12)
    plt.yticks(np.arange(0, basis_data[basis_idx].shape[0], 1), fontsize=12)
    plt.show()