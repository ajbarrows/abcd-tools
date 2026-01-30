"""Visualization utilities for neuroimaging data.

This module provides functions for visualizing brain surfaces, ROIs, and atlases.
Requires nilearn for surface plotting functionality.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

try:
    from nilearn import datasets, plotting
except ImportError:
    raise ImportError(
        "nilearn is required for visualization. "
        "Install it with: pip install abcd-tools[viz]"
    )


def plot_surface_roi(
    mask: Any,
    label: str,
    hemi: str,
    fsaverage: dict,
    ax: plt.Axes,
    cmap: str = "Reds",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Plot a single ROI on a surface mesh.

    Args:
        mask: Binary mask array for the ROI
        label (str): Label/name of the ROI
        hemi (str): Hemisphere to plot ('left' or 'right')
        fsaverage (dict): fsaverage surface data from nilearn
        ax (plt.Axes): Matplotlib axis to plot on
        cmap (str, optional): Colormap name. Defaults to 'Reds'.
        vmin (float, optional): Minimum value for colormap. Defaults to 0.
        vmax (float, optional): Maximum value for colormap. Defaults to 1.
    """
    plotting.plot_surf_roi(
        fsaverage[f"pial_{hemi}"],
        roi_map=mask,
        title=f"{label} ({hemi})",
        axes=ax,
        hemi=hemi,
        view="lateral",
        bg_map=fsaverage[f"sulc_{hemi}"],
        bg_on_data=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )


def generate_roi_table(
    atlas: dict,
    descriptions: dict,
    output_path: str | Path = "./",
    dpi: int = 100,
    figsize: tuple = (10, 3),
) -> pd.DataFrame:
    """Generate an HTML table with surface visualizations for each ROI in an atlas.

    Creates individual PNG images for each ROI (showing both hemispheres) and
    compiles them into an HTML table with ROI names and embedded images.

    Args:
        atlas (dict): Atlas dictionary with 'lut', 'map_left', and 'map_right' keys.
            Typically from nilearn.datasets.fetch_atlas_surf_* functions.
        descriptions (dict): Dictionary mapping ROI indices to full description names.
        output_path (str | Path, optional): Directory to save images and HTML table.
            Defaults to "./".
        dpi (int, optional): Resolution for saved images. Defaults to 100.
        figsize (tuple, optional): Figure size (width, height) in inches.
            Defaults to (10, 3).

    Returns:
        pd.DataFrame: DataFrame with 'ROI' and 'Image' columns used to generate
            the HTML table.

    Example:
        >>> from nilearn import datasets
        >>> from abcd_tools.utils.config_loader import load_yaml
        >>> from abcd_tools.viz import generate_roi_table
        >>>
        >>> # Load atlas and descriptions
        >>> atlas = datasets.fetch_atlas_surf_destrieux()
        >>> mappings = load_yaml("conf/mappings.yaml")
        >>> descriptions = mappings['destrieux_descriptions']
        >>>
        >>> # Generate ROI table
        >>> df = generate_roi_table(atlas, descriptions, output_path="./output/")
    """
    output_path = Path(output_path)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Filter lookup table to exclude Unknown regions
    lut = atlas["lut"].query("name != 'Unknown'")
    indices = lut["index"]
    labels = lut["name"]

    # Fetch fsaverage surface
    fsaverage = datasets.fetch_surf_fsaverage()

    rows = []

    for index, label in zip(indices, labels):
        # Create binary masks for each hemisphere
        left_mask = (atlas["map_left"] == index).astype(int)
        right_mask = (atlas["map_right"] == index).astype(int)

        # Create figure with two subplots for hemispheres
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, subplot_kw={"projection": "3d"}
        )

        # Plot both hemispheres
        plot_surface_roi(left_mask, label, "left", fsaverage, axes[0])
        plot_surface_roi(right_mask, label, "right", fsaverage, axes[1])

        # Save figure
        img_path = images_dir / f"roi_{index}.png"
        plt.savefig(img_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        plt.close()

        # Create HTML table entry
        long_name = descriptions[index]
        img_path_rel = f"images/roi_{index}.png"
        rows.append(
            {"ROI": long_name, "Image": f'<img src="{img_path_rel}" width="600">'}
            )

    # Create DataFrame and save as HTML
    df = pd.DataFrame(rows)
    html_path = output_path / "roi_table.html"
    df.to_html(html_path, escape=False, index=False)

    print(f"ROI table saved to: {html_path}")
    print(f"Generated {len(rows)} ROI images in: {images_dir}")

    return df



def plot_brain_views(lh_data, rh_data, ax, fsaverage,
                     vmin=None, vmax=None, cmap='RdBu_r',
                     colorbar=True):
    # Calculate vmin/vmax if not provided
    if vmin is None or vmax is None:
        max_val = max(abs(lh_data).max(), abs(rh_data).max())
        vmin = -max_val if vmin is None else vmin
        vmax = max_val if vmax is None else vmax

    # Create 2x2 grid for brain plots - larger and closer together
    ax1 = ax.inset_axes([0, 0.5, 0.45, 0.5], projection='3d')  # top-left: LH lateral
    ax2 = ax.inset_axes([0.45, 0.5, 0.45, 0.5], projection='3d')  # top-right: RH lat.
    ax3 = ax.inset_axes([0, 0, 0.45, 0.5], projection='3d')  # bottom-left: LH lateral
    ax4 = ax.inset_axes([0.45, 0, 0.45, 0.5], projection='3d')  # bottom-right: RH med.
    cbar_ax = ax.inset_axes([0.92, 0.1, 0.02, 0.8])  # colorbar on right (narrower)
    ax.set_axis_off()

    plotting.plot_surf_stat_map(fsaverage['pial_left'], lh_data,
                                hemi='left', view='lateral', axes=ax1,
                                vmin=vmin, vmax=vmax, cmap=cmap, colorbar=False)
    plotting.plot_surf_stat_map(fsaverage['pial_right'], rh_data,
                                hemi='right', view='lateral', axes=ax2,
                                vmin=vmin, vmax=vmax, cmap=cmap, colorbar=False)

    plotting.plot_surf_stat_map(fsaverage['pial_left'], lh_data,
                                hemi='left', view='medial', axes=ax3,
                                vmin=vmin, vmax=vmax, cmap=cmap, colorbar=False)

    plotting.lot_surf_stat_map(fsaverage['pial_right'], rh_data,
                                          hemi='right', view='medial', axes=ax4,
                                          vmin=vmin, vmax=vmax, cmap=cmap,
                                          colorbar=False)

    if colorbar:
        # Add colorbar manually with ScalarMappable
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, cax=cbar_ax)
