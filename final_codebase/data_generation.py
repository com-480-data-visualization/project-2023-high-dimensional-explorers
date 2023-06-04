# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Sampling scheme for the data used in the visualization."""
import numpy as np
import scipy.stats.qmc
import torch


def build_pointset(
    num_points: int,
    pointset_type: str,
    dim_in: int,
    lower: np.array,
    upper: np.array,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build a pointset.

    The pointset_type can be: None, "sobol", "halton". None means pseudorandom.

    can be changed by hydra config
    `python main_app.py pointset_type=sobol`
    """
    if pointset_type is None:
        uc = np.random.rand(num_points, dim_in)
    elif "sobol" == pointset_type:
        soboleng = torch.quasirandom.SobolEngine(dimension=dim_in, scramble=True)
        uc = soboleng.draw(num_points, dtype=dtype)
    elif "halton" == pointset_type:
        halton_sampler = scipy.stats.qmc.Halton(d=dim_in, scramble=True)
        uc = halton_sampler.random(num_points)
    else:
        raise ValueError(f"Do not know about pointset_type = {pointset_type}")
    pointset = lower + uc * (upper - lower)
    return pointset
