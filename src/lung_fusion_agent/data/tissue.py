from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TissueMaskConfig:
    saturation_threshold: float
    minimum_value: float
    maximum_value: float
    minimum_component_fraction: float
    morphology_kernel_size: int
    opening_iterations: int
    closing_iterations: int


def calculate_tissue_mask(
    rgb: np.ndarray,
    config: TissueMaskConfig,
) -> np.ndarray:
    """Create a binary tissue mask from an RGB uint8 image."""
    normalized = rgb.astype(np.float32) / 255.0

    maximum = normalized.max(axis=2)
    minimum = normalized.min(axis=2)

    saturation = (maximum - minimum) / (maximum + 1e-6)
    value = maximum

    mask = (
        (saturation >= config.saturation_threshold)
        & (value >= config.minimum_value)
        & (value <= config.maximum_value)
    )

    structure = np.ones(
        (
            config.morphology_kernel_size,
            config.morphology_kernel_size,
        ),
        dtype=bool,
    )

    mask = ndimage.binary_opening(
        mask,
        structure=structure,
        iterations=config.opening_iterations,
    )
    mask = ndimage.binary_closing(
        mask,
        structure=structure,
        iterations=config.closing_iterations,
    )

    labels, component_count = ndimage.label(mask)

    if component_count:
        component_sizes = np.bincount(labels.ravel())
        minimum_size = max(
            64,
            int(mask.size * config.minimum_component_fraction),
        )

        keep = component_sizes >= minimum_size
        keep[0] = False
        mask = keep[labels]

    return mask
