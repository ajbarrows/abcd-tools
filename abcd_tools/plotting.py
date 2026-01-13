"""Visualization utilities for neuroimaging data.

This module provides functions for visualizing brain surfaces, ROIs, and atlases.
Requires nilearn for surface plotting functionality.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from nilearn import datasets, plotting
except ImportError:
    raise ImportError(
        "nilearn is required for plotting. "
        "Install it with: pip install abcd-tools[plotting]"
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
        img_html = f'<img src="{img_path_rel}" width="600">'
        rows.append({"ROI": long_name, "Image": img_html})

    # Create DataFrame and save as HTML
    df = pd.DataFrame(rows)
    html_path = output_path / "roi_table.html"
    df.to_html(html_path, escape=False, index=False)

    print(f"ROI table saved to: {html_path}")
    print(f"Generated {len(rows)} ROI images in: {images_dir}")

    return df
