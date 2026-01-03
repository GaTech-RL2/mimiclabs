import numpy as np
from transforms3d import euler
import matplotlib.colors as mcolors

from ..utils import disable_module_import

with disable_module_import("libero", "libero", "envs"):
    from libero.libero.envs.utils import *


def convert_spherical_to_pos_quat(r_theta_phi):
    """
    Converts spherical coordinates (physics convention) into Euclidean position
    and quaternion rotation, such that -z points at the origin.
    """
    r, theta, phi = r_theta_phi
    pos = [
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]
    euler_zxy = (np.pi / 2 + phi, theta, 0)
    quat_wxyz = euler.euler2quat(*euler_zxy, axes="rzxy")
    return pos, quat_wxyz


def sample_hsv_from_hsv_ranges(hsv_ranges):
    """
    Samples a random HSV color within the given ranges.

    Args:
        hsv_ranges (list): List of 3 tuples, each containing (min, max) for H, S, and V.

    Returns:
        tuple: A tuple representing the sampled HSV color.
    """
    hsv_range_choice = np.random.choice(range(len(hsv_ranges)))
    hsv_range = hsv_ranges[hsv_range_choice]
    # Sample all HSV components from the range
    hue = np.random.choice(range(hsv_range[0], hsv_range[3] + 1))
    sat = np.random.choice(range(hsv_range[1], hsv_range[4] + 1))
    val = np.random.choice(range(hsv_range[2], hsv_range[5] + 1))
    return (hue, sat, val)


def hsv_to_rgba(h, s, v, alpha=1.0):
    """
    Converts HSV values to RGBA values (0-1 range) suitable for MuJoCo.

    Args:
        h (float): Hue value (0 to 179).
        s (float): Saturation value (0 to 255).
        v (float): Value/Brightness value (0 to 255).
        alpha (float): Alpha/Transparency value (0.0 to 1.0).

    Returns:
        tuple: A tuple (r, g, b, a) with values in the range [0.0, 1.0].
    """
    # Normalize HSV values to 0-1 range for matplotlib
    h_normalized = h / 179.0
    s_normalized = s / 255.0
    v_normalized = v / 255.0

    # Matplotlib's function takes an array-like input of shape (..., 3)
    hsv_color = np.array([h_normalized, s_normalized, v_normalized])

    # Convert hsv to rgb (returns a numpy array of shape (..., 3))
    rgb_color = mcolors.hsv_to_rgb(hsv_color)

    # Combine RGB with the alpha channel
    rgba_color = np.append(rgb_color, alpha)

    return tuple(rgba_color)
