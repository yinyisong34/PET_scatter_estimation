import numpy as np
import matplotlib.pyplot as plt

def sigma_0123(sigma_negml):
    sigma0 = sigma_negml[0]
    sigma1 = sigma_negml[1]
    sigma2 = sigma_negml[2]
    sigma3 = sigma_negml[3]

    sigma_scatter = sigma1 + sigma2 + sigma3

    return sigma0, sigma1, sigma2, sigma3, sigma_scatter

def plot_negml_results(
    scatter_sinogram,
    nonscatter_sinogram,
    sigma_scatter_negml,
    sigma0_negml,
    basis_type,
    nonscatter_type,
    scatter_type,
    view_idx=16
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Make labels/titles nicer
    basis_label = {
        "global": "Global basis",
        "multi-region": "Multi-region basis"
    }.get(basis_type, basis_type)

    nonscatter_label = {
        "predicted": "Predicted non-scatter",
        "data-normalised": "Data-normalised non-scatter"
    }.get(nonscatter_type, nonscatter_type)

    scatter_label = {
        "predicted": "Predicted scatter",
        "data-normalised": "Data-normalised scatter"
    }.get(scatter_type, scatter_type)

    # 1. Scatter at selected view
    axes[0, 0].plot(
        np.mean(scatter_sinogram[0, :, view_idx, :], axis=0),
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].plot(
        np.mean(sigma_scatter_negml[0, :, view_idx, :], axis=0),
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 0].set_title(
        f"Scatter: view {view_idx}\n{scatter_label}"
    )

    # 2. Scatter averaged across axial and views
    axes[0, 1].plot(
        np.mean(scatter_sinogram[0], axis=(0, 1)),
        label="True scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].plot(
        np.mean(sigma_scatter_negml[0], axis=(0, 1)),
        label="NEGML scatter",
        marker="o",
        ms=4
    )

    axes[0, 1].set_title(
        f"Scatter: averaged over axial and view\n{scatter_label}"
    )

    # 3. Non-scatter at selected view
    axes[1, 0].plot(
        np.mean(nonscatter_sinogram[0, :, view_idx, :], axis=0),
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].plot(
        np.mean(sigma0_negml[0, :, view_idx, :], axis=0),
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 0].set_title(
        f"Non-scatter: view {view_idx}\n{nonscatter_label}"
    )

    # 4. Non-scatter averaged across axial and views
    axes[1, 1].plot(
        np.mean(nonscatter_sinogram[0], axis=(0, 1)),
        label="True non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].plot(
        np.mean(sigma0_negml[0], axis=(0, 1)),
        label="NEGML non-scatter",
        marker="o",
        ms=4
    )

    axes[1, 1].set_title(
        f"Non-scatter: averaged over axial and view\n{nonscatter_label}"
    )

    for ax in axes.flat:
        ax.legend(frameon=False)
        ax.tick_params(axis="both", labelsize=11)

    fig.suptitle(
        f"NEGML results — {basis_label}\n"
        f"{nonscatter_label} + {scatter_label}",
        fontsize=16
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)