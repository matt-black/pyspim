"""
Utility functions for the napari-pyspim plugin.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import numpy as np


class AffineComponents(NamedTuple):
    """Decomposed components of a 4x4 affine transformation matrix.

    Attributes:
        translation: 3-element array of translation offsets [tx, ty, tz].
        rotation_matrix: 3x3 pure rotation matrix (det = +1).
        euler_angles: 3-element array [alpha, beta, gamma] in radians,
            using yaw-pitch-roll (ZYX intrinsic) convention.
        scale: 3-element array of per-axis scale factors [sx, sy, sz].
        shear: 3-element array of upper-triangular shear factors [h_xy, h_xz, h_yz].
    """

    translation: np.ndarray
    rotation_matrix: np.ndarray
    euler_angles: np.ndarray
    scale: np.ndarray
    shear: np.ndarray


def decompose_affine_matrix(matrix: np.ndarray) -> AffineComponents:
    """Decompose a 4x4 affine transformation matrix into translation, rotation, scale, and shear.

    Uses Cholesky decomposition of the linear part to separate rotation, scale, and shear
    components. Euler angles are extracted from the rotation matrix using the yaw-pitch-roll
    (ZYX intrinsic) convention matching the existing pyspim matrix constructors.

    Args:
        matrix: A 4x4 affine transformation matrix (NumPy array).

    Returns:
        AffineComponents named tuple containing:
            - translation: 3-element array of translation offsets [tx, ty, tz]
            - rotation_matrix: 3x3 pure rotation matrix
            - euler_angles: 3-element array [alpha, beta, gamma] in radians
            - scale: 3-element array of per-axis scale factors [sx, sy, sz]
            - shear: 3-element array of upper-triangular shear factors [h_xy, h_xz, h_yz]

    Raises:
        ValueError: If input is not a 4x4 matrix.
        np.linalg.LinAlgError: If Cholesky decomposition fails
            (matrix not positive definite).

    Example:
        >>> matrix = np.eye(4)
        >>> comp = _decompose_affine_matrix(matrix)
        >>> comp.translation
        array([0., 0., 0.])
        >>> comp.scale
        array([1., 1., 1.])
    """
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 affine matrix, got shape {matrix.shape}"
        )

    # Step 1: Extract translation from last column
    translation = matrix[:3, 3].copy()

    # Step 2: Extract the 3x3 linear part
    RZS = matrix[:3, :3].copy()

    # Step 3: Cholesky decomposition of RZS.T @ RZS
    # RZS.T @ RZS is symmetric positive semi-definite for valid affine matrices
    RZS_T_RZS = RZS.T @ RZS
    ZS = np.linalg.cholesky(RZS_T_RZS).T  # Upper triangular

    # Step 4: Extract scale (diagonal) and shear (off-diagonal)
    scale = np.diag(ZS).copy()
    shear_matrix = ZS / scale[:, np.newaxis]
    shear = shear_matrix[np.triu(np.ones((3, 3)), 1).astype(bool)]

    # Step 5: Compute rotation matrix
    R = RZS @ np.linalg.inv(ZS)

    # Step 6: Handle negative determinant (reflection)
    if np.linalg.det(R) < 0:
        scale[0] *= -1
        ZS[0, :] *= -1
        R = RZS @ np.linalg.inv(ZS)

    # Step 7: Extract Euler angles (yaw-pitch-roll / ZYX intrinsic convention)
    # Matches the convention in pyspim._matrix.rotation_matrix():
    #   R = yaw(alpha) @ pitch(beta) @ roll(gamma)
    # From the composition:
    #   R[2,0] = -sin(beta)
    #   R[1,0]/R[0,0] = tan(alpha)
    #   R[2,1]/R[2,2] = tan(gamma)
    euler_angles = _extract_euler_angles(R)

    return AffineComponents(
        translation=translation,
        rotation_matrix=R,
        euler_angles=euler_angles,
        scale=scale,
        shear=shear,
    )


def _extract_euler_angles(R: np.ndarray) -> np.ndarray:
    """Extract yaw-pitch-roll Euler angles from a 3x3 rotation matrix.

    Uses the ZYX intrinsic convention matching pyspim._matrix.rotation_matrix():
        R = yaw(alpha) @ pitch(beta) @ roll(gamma)

    Args:
        R: 3x3 rotation matrix.

    Returns:
        3-element array [alpha, beta, gamma] in radians.
    """
    # beta from R[2, 0] = -sin(beta)
    beta = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

    # Check for gimbal lock: |cos(beta)| near zero
    cos_beta = np.cos(beta)
    epsilon = 1e-12

    if np.abs(cos_beta) < epsilon:
        # Gimbal lock — beta ≈ ±π/2
        # alpha is arbitrary; set to 0 and solve for gamma
        alpha = 0.0
        # When cos(beta) ≈ 0, R[0,1] = -sign(beta)*sin(gamma), R[0,2] = sign(beta)*cos(gamma)
        sign_beta = -np.sign(R[2, 0])  # sin(beta) = -R[2,0]
        gamma = np.arctan2(-sign_beta * R[0, 1], sign_beta * R[0, 2])
    else:
        # Normal case
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = np.arctan2(R[2, 1], R[2, 2])

    return np.array([alpha, beta, gamma])


def save_workflow_parameters(output_path: str, parameters: Dict[str, Any]) -> None:
    """Save workflow parameters to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(parameters, f, indent=2, default=str)


def load_workflow_parameters(input_path: str) -> Dict[str, Any]:
    """Load workflow parameters from a JSON file."""
    with open(input_path) as f:
        return json.load(f)


def get_layer_metadata(viewer, layer_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata from a napari layer."""
    try:
        layer = viewer.layers[layer_name]
        return layer.metadata
    except (KeyError, AttributeError):
        return None


def format_memory_usage(array: np.ndarray) -> str:
    """Format memory usage of an array in human-readable format."""
    size_bytes = array.size * array.itemsize
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def validate_data_path(path: str) -> bool:
    """Validate that a data path exists and contains expected files."""
    if not os.path.exists(path):
        return False

    # Check for common μManager file patterns
    expected_files = ["metadata.txt", "Pos0", "Pos1"]
    path_obj = Path(path)

    # Check if any expected files exist
    for file_pattern in expected_files:
        if list(path_obj.glob(f"*{file_pattern}*")):
            return True

    # If no expected files, check if it's a directory with subdirectories
    return any(path_obj.iterdir())


def create_output_directory(base_path: str, acquisition_name: str) -> str:
    """Create an output directory for processed data."""
    output_dir = os.path.join(base_path, f"{acquisition_name}_processed")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_default_psf_paths() -> tuple[str, str]:
    """Get default PSF paths based on common locations."""
    common_paths = [
        "/scratch/gpfs/SHAEVITZ/dispim/extract_spindles",
        "/projects/SHAEVITZ/dispim/psfs",
        os.path.expanduser("~/dispim/psfs"),
    ]

    for path in common_paths:
        psf_a = os.path.join(path, "PSFA_500.npy")
        psf_b = os.path.join(path, "PSFB_500.npy")
        if os.path.exists(psf_a) and os.path.exists(psf_b):
            return psf_a, psf_b

    return "", ""


def estimate_processing_time(shape: tuple, iterations: int = 20) -> str:
    """Estimate processing time based on data size and parameters."""
    total_pixels = np.prod(shape)

    # Rough estimates based on typical processing speeds
    # These are very approximate and depend on hardware
    if total_pixels < 10**7:  # < 10M pixels
        time_minutes = iterations * 0.1
    elif total_pixels < 10**8:  # < 100M pixels
        time_minutes = iterations * 0.5
    elif total_pixels < 10**9:  # < 1B pixels
        time_minutes = iterations * 2
    else:  # > 1B pixels
        time_minutes = iterations * 10

    if time_minutes < 60:
        return f"~{time_minutes:.1f} minutes"
    else:
        hours = time_minutes / 60
        return f"~{hours:.1f} hours"


def format_shape_string(shape: tuple) -> str:
    """Format array shape as a readable string."""
    if len(shape) == 3:
        return f"{shape[0]}×{shape[1]}×{shape[2]} (Z×Y×X)"
    elif len(shape) == 2:
        return f"{shape[0]}×{shape[1]} (Y×X)"
    else:
        return str(shape)
