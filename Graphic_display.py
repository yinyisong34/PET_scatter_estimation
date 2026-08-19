# SPDX-License-Identifier: Apache-2.0
"""Plotting and labelling utilities for the EBSE/NEGML analysis.

The functions in this module split fitted NEGML components, generate labels
from analysis settings, compare fitted and ground-truth sinograms, and
visualise individual energy-basis images."""

import numpy as np
import matplotlib.pyplot as plt

def sigma_0123(sigma_negml):
    """Split a four-component NEGML result and form the total scatter estimate.
    
    Parameters
    ----------
    sigma_negml : numpy.ndarray
        Coefficient array whose first axis contains the four fitted components
        C0, C1, C2, and C3.
    
    Returns
    -------
    sigma0, sigma1, sigma2, sigma3 : numpy.ndarray
        Individual fitted coefficient arrays.
    sigma_scatter : numpy.ndarray
        Combined scatter estimate ``sigma1 + sigma2 + sigma3``.
    """
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
    """Generate descriptive plot labels from the NEGML configuration.
    
    Parameters
    ----------
    basis_function : callable
        Basis-construction function used in the fit.
    U_true_or_predict : str
        Source flag for the non-scatter PDF, typically ``"true"`` or
        ``"predicted"``.
    S_true_or_predict : str
        Source flag for the scatter PDF, typically ``"true"`` or
        ``"predicted"``.
    pixel_width, pixel_length : int
        Spatial region dimensions used for basis estimation.
    sino_shape : tuple
        Spatial sinogram shape. The final two entries are interpreted as the
        view and tangential dimensions.
    
    Returns
    -------
    U_label : str
        Human-readable non-scatter PDF label.
    S_label : str
        Human-readable scatter PDF label.
    basis_label : str
        Human-readable basis-construction label.
    spatial_label : str
        Label identifying the global or multi-region spatial model.
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


def plot_negml_results_avg_z(
    scatter_sinogram,
    nonscatter_sinogram,
    sigma_scatter_negml,
    sigma0_negml,
    U_label,
    S_label,
    basis_label,
    spatial_label,
    axial_fit_mode="avg",
    view_idx=16
):
    """Plot fitted and ground-truth scatter/non-scatter sinogram profiles.

    A 2-by-2 figure is produced containing scatter and non-scatter comparisons
    at one selected view and after averaging over the axial and view axes.

    The plotted profiles are averaged over the axial direction for
    visualisation. This plotting average is separate from the axial treatment
    used during NEGML fitting, which is specified by ``axial_fit_mode``.

    Parameters
    ----------
    scatter_sinogram : numpy.ndarray
        Ground-truth total scatter sinogram.
    nonscatter_sinogram : numpy.ndarray
        Ground-truth non-scatter sinogram.
    sigma_scatter_negml : numpy.ndarray
        NEGML-estimated total scatter sinogram.
    sigma0_negml : numpy.ndarray
        NEGML-estimated non-scatter (C0) sinogram.
    U_label, S_label : str
        Labels describing the PDF sources.
    basis_label : str
        Label describing the basis construction.
    spatial_label : str
        Label describing the spatial basis model.
    axial_fit_mode : str, optional
        Describes how the axial dimension was treated during NEGML fitting.

        - ``"all"``:
          NEGML is fitted independently at each axial position.

        - ``"avg"``:
          Energy histograms are first averaged over the axial direction,
          and NEGML is fitted to the axially averaged data.

        Default is ``"avg"``.
    view_idx : int, optional
        View index shown in the selected-view panels. Default is 16.

    Returns
    -------
    None
        The figure is displayed and then closed.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---------------------------------------------------------
    # Label describing how the NEGML model itself was fitted
    # ---------------------------------------------------------

    if axial_fit_mode == "all":
        axial_model_label = (
            "Axial fitting: independent NEGML fit at each z position"
        )

    elif axial_fit_mode == "avg":
        axial_model_label = (
            "Axial fitting: energy data averaged over z before NEGML"
        )

    else:
        raise ValueError(
            "axial_fit_mode must be either 'all' or 'avg'."
        )


    # ---------------------------------------------------------
    # 1. Scatter at selected view
    # ---------------------------------------------------------
    # The model may or may not have used axial averaging during fitting.
    # Here, however, the axial dimension is averaged only for plotting,
    # so that the result can be displayed as a 1D tangential profile.

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
        f"Scatter: view {view_idx} (axial mean)"
    )


    # ---------------------------------------------------------
    # 2. Scatter averaged over axial position and view
    # ---------------------------------------------------------

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
        "Scatter: mean over axial position and view"
    )


    # ---------------------------------------------------------
    # 3. Non-scatter at selected view
    # ---------------------------------------------------------

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
        f"Non-scatter: view {view_idx} (axial mean)"
    )


    # ---------------------------------------------------------
    # 4. Non-scatter averaged over axial position and view
    # ---------------------------------------------------------

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
        "Non-scatter: mean over axial position and view"
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
        ax.set_xlabel("Tangential position")


    fig.suptitle(
        "NEGML results\n"
        f"{spatial_label}\n"
        f"{basis_label}\n"
        f"{axial_model_label}\n"
        f"U(E): {U_label}   |   S(E): {S_label}",
        fontsize=15
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)



def plot_negml_results_at_z(
    scatter_sinogram,
    nonscatter_sinogram,
    sigma_scatter_negml,
    sigma0_negml,
    U_label,
    S_label,
    basis_label,
    spatial_label,
    axial_fit_mode="all",
    view_idx=16,
    z_idx=0
):
    """Plot fitted and ground-truth scatter/non-scatter profiles at one axial position.

    A 2-by-2 figure is produced containing scatter and non-scatter comparisons
    at one selected axial position. The left-hand panels show one selected
    view, while the right-hand panels average over all view angles at the
    same axial position.

    Unlike ``plot_negml_results``, no averaging over the axial direction is
    performed for visualisation. This allows variation between individual
    axial positions to be examined.

    Parameters
    ----------
    scatter_sinogram : numpy.ndarray
        Ground-truth total scatter sinogram.
    nonscatter_sinogram : numpy.ndarray
        Ground-truth non-scatter sinogram.
    sigma_scatter_negml : numpy.ndarray
        NEGML-estimated total scatter sinogram.
    sigma0_negml : numpy.ndarray
        NEGML-estimated non-scatter (C0) sinogram.
    U_label, S_label : str
        Labels describing the PDF sources.
    basis_label : str
        Label describing the basis construction.
    spatial_label : str
        Label describing the spatial basis model.
    axial_fit_mode : str, optional
        Describes how the axial dimension was treated during NEGML fitting.

        - ``"all"``:
          NEGML is fitted independently at each axial position.

        - ``"avg"``:
          Energy histograms are first averaged over the axial direction,
          and NEGML is fitted to the axially averaged data.

        Default is ``"all"``.
    view_idx : int, optional
        View index shown in the selected-view panels. Default is 16.
    z_idx : int, optional
        Axial position shown in all four panels. Default is 0.

    Returns
    -------
    None
        The figure is displayed and then closed.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if axial_fit_mode == "all":
        axial_model_label = (
            "Axial fitting: independent NEGML fit at each z position"
        )

    elif axial_fit_mode == "avg":
        axial_model_label = (
            "Axial fitting: energy data averaged over z before NEGML"
        )

    else:
        raise ValueError(
            "axial_fit_mode must be either 'all' or 'avg'."
        )


    # 1. Scatter at the selected axial position and view
    axes[0, 0].plot(
        scatter_sinogram[0, z_idx, view_idx, :],
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].plot(
        sigma_scatter_negml[0, z_idx, view_idx, :],
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].set_title(
        f"Scatter: axial {z_idx}, view {view_idx}"
    )


    # 2. Scatter at the selected axial position, averaged over view
    axes[0, 1].plot(
        np.mean(
            scatter_sinogram[0, z_idx],
            axis=0
        ),
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].plot(
        np.mean(
            sigma_scatter_negml[0, z_idx],
            axis=0
        ),
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].set_title(
        f"Scatter: axial {z_idx}, mean over view"
    )


    # 3. Non-scatter at the selected axial position and view
    axes[1, 0].plot(
        nonscatter_sinogram[0, z_idx, view_idx, :],
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].plot(
        sigma0_negml[0, z_idx, view_idx, :],
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].set_title(
        f"Non-scatter: axial {z_idx}, view {view_idx}"
    )


    # 4. Non-scatter at the selected axial position, averaged over view
    axes[1, 1].plot(
        np.mean(
            nonscatter_sinogram[0, z_idx],
            axis=0
        ),
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].plot(
        np.mean(
            sigma0_negml[0, z_idx],
            axis=0
        ),
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].set_title(
        f"Non-scatter: axial {z_idx}, mean over view"
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
        ax.set_xlabel("Tangential position")


    fig.suptitle(
        "NEGML results\n"
        f"{spatial_label}\n"
        f"{basis_label}\n"
        f"{axial_model_label}\n"
        f"U(E): {U_label}   |   S(E): {S_label}",
        fontsize=15
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)



def plot_basis(
    pixel,
    basis_type,
    basis_idx,
    pdf_source=None,
    resolution=10,
    basis_data=None
):
    """Plot one 2D basis component with an automatically generated title.

    Parameters
    ----------
    pixel : tuple of int
        View--tangential pixel location, for example ``(16, 10)``.
    basis_type : {"shared", "separate", "GT_2D", "ground_truth", "ground-truth"}
        Basis representation to label.
    basis_idx : int
        Basis component index: 0, 1, 2, or 3.
    pdf_source : {"true", "predicted"}, optional
        Source of the 1D PDFs used to construct the factorised basis.

        - ``"true"``:
          Ground-truth 1D PDFs are used.
        - ``"predicted"``:
          Predicted 1D PDFs are used.

        This argument is required for ``"shared"`` and ``"separate"`` basis
        types, but is not used for ``"GT_2D"`` because the ground-truth 2D
        basis has no predicted equivalent.
    resolution : int, optional
        Resolution value used only to construct the fallback variable name when
        ``basis_data`` is not supplied. Default is 10.
    basis_data : array-like, optional
        Basis array to plot directly. If omitted, the function searches the
        caller's global namespace for the corresponding basis variable.

    Returns
    -------
    None
        Displays the requested basis image.

    Raises
    ------
    ValueError
        If ``basis_type``, ``basis_idx``, or ``pdf_source`` is invalid, or if
        the expected basis variable cannot be found when ``basis_data`` is
        omitted.
    """

    i, j = pixel

    # ---------------------------------------------------------
    # Basis-type aliases
    # ---------------------------------------------------------

    basis_type_alias = {
        "shared": "shared",
        "separate": "separate",
        "GT_2D": "GT_2D",
        "ground_truth": "GT_2D",
        "ground-truth": "GT_2D"
    }

    if basis_type not in basis_type_alias:
        raise ValueError(
            "basis_type must be one of: 'shared', 'separate', 'GT_2D'"
        )

    basis_type = basis_type_alias[basis_type]


    # ---------------------------------------------------------
    # Validate PDF source
    # ---------------------------------------------------------

    if basis_type in ["shared", "separate"]:

        if pdf_source not in ["true", "predicted"]:
            raise ValueError(
                "For 'shared' or 'separate' basis types, "
                "pdf_source must be either 'true' or 'predicted'."
            )

    else:
        # GT_2D is already constructed directly from ground-truth
        # 2D energy histograms, so there is no predicted-PDF version.
        pdf_source = None


    # ---------------------------------------------------------
    # Title prefix and formula mapping
    # ---------------------------------------------------------

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
            r"B_0(E_1,E_2)",
            r"B_1(E_1,E_2)",
            r"B_2(E_1,E_2)",
            r"B_3(E_1,E_2)"
        ]
    }

    if basis_idx not in [0, 1, 2, 3]:
        raise ValueError(
            "basis_idx must be 0, 1, 2, or 3"
        )


    # ---------------------------------------------------------
    # Locate basis data if it was not supplied directly
    # ---------------------------------------------------------

    if basis_data is None:
        import inspect

        if basis_type == "GT_2D":

            var_name = (
                f"basis_resolution_{resolution}_"
                f"pixel_{i}_{j}_GT_2D"
            )

        else:

            var_name = (
                f"basis_resolution_{resolution}_"
                f"pixel_{i}_{j}_{basis_type}_{pdf_source}"
            )

        # Get the namespace of the notebook/script that called this function
        caller_frame = inspect.currentframe().f_back
        caller_namespace = caller_frame.f_globals

        if var_name not in caller_namespace:
            raise ValueError(
                f"Cannot find variable '{var_name}' "
                "in the caller namespace."
            )

        basis_data = caller_namespace[var_name]


    # ---------------------------------------------------------
    # Construct title
    # ---------------------------------------------------------

    if basis_type == "GT_2D":

        title = (
            rf"{title_prefix[basis_type]} "
            rf"${title_formula[basis_type][basis_idx]}$ "
            "\n"
            rf"at pixel ({i}, {j})"
        )

    else:

        pdf_source_label = {
            "true": "ground-truth PDFs",
            "predicted": "predicted PDFs"
        }

        title = (
            rf"{title_prefix[basis_type]} "
            rf"${title_formula[basis_type][basis_idx]}$ "
            "\n"
            rf"using {pdf_source_label[pdf_source]} "
            rf"at pixel ({i}, {j})"
        )


    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.imshow(
        basis_data[basis_idx]
    )

    plt.colorbar()

    plt.title(
        title,
        fontsize=15,
        pad=20
    )

    plt.xticks(
        np.arange(
            0,
            basis_data[basis_idx].shape[1],
            1
        ),
        fontsize=12
    )

    plt.yticks(
        np.arange(
            0,
            basis_data[basis_idx].shape[0],
            1
        ),
        fontsize=12
    )

    plt.show()