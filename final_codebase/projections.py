import functools
import math

import numpy as np
import torch


def persp_proj_down_to_dim_from_end(
    points: torch.Tensor, target_dim: int
) -> torch.Tensor:
    assert 2 == points.ndim
    n = points.shape[1]

    projed = points
    for idx in range(n - target_dim):
        projed = _proj_down_from_end_persp(projed)
    return projed


def _proj_down_from_end_persp(points: torch.Tensor) -> torch.Tensor:
    assert 2 == points.ndim
    n = points.shape[1]
    p = 1.0
    projed = points[:, :-1] / (p * n - points[:, -1]).reshape(-1, 1)
    return projed


def get_proj32t_mat(dim=3) -> np.ndarray:
    proj32t_mat = np.eye(dim)[:-1, :].T
    return proj32t_mat


def all_angular_proj(points: np.array, angle: float, target_dim: int) -> np.array:
    assert 2 == points.ndim
    n = points.shape[1]

    projed = points
    for idx in range(n - target_dim):
        projed = _proj_down_from_end_at_angle(projed, angle)
    return projed


def _proj_down_from_end_at_angle(points: np.array, angle: float) -> np.array:
    assert 2 == points.ndim
    n = points.shape[1]
    assert n > 2

    post_mult = rotation_matrix_prod_fast(angle) @ get_proj32t_mat()
    projected_points = points[:, -3:] @ post_mult
    projed = np.concatenate((points[:, :-3], projected_points), 1)
    return projed


def rotation_matrix_prod_fast(angle: float) -> np.array:
    rmp = (
        np.array(
            [
                [math.cos(angle), +math.sin(angle), 0],
                [-math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        @ np.array(
            [
                [math.cos(angle), 0, -math.sin(angle)],
                [0, 1, 0],
                [+math.sin(angle), 0, math.cos(angle)],
            ]
        )
        @ np.array(
            [
                [1, 0, 0],
                [0, math.cos(angle), +math.sin(angle)],
                [0, -math.sin(angle), math.cos(angle)],
            ]
        )
    )
    return rmp


def rotation_matrix_prod(angle: float) -> np.array:
    rotation_z = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotation_y = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
    )
    rmp = rotation_z.T @ rotation_y.T @ rotation_x.T
    return rmp


def parameterized_linear() -> np.array:
    projed = []
    return projed


if __name__ == "__main__":
    angle = 0.43

    rmp_fast = rotation_matrix_prod_fast(angle)
    rmp = rotation_matrix_prod(angle)

    np.testing.assert_allclose(rmp, rmp_fast)
