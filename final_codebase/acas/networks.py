import sys
import os
import time
import functools
import datetime as dt
import logging
from typing import Any, Dict, List, Tuple
from collections import Counter

import omegaconf

# warnings.filterwarnings("ignore", category=Warning)
# warnings.filterwarnings("error")
import hydra
import torch.optim
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import h5py


def build_relu_layers(input_dim: int,
                      hidden_layer_widths: List[int],
                      output_dim: int,
                      include_bias: bool) -> List[torch.nn.Module]:
    num_layers = len(hidden_layer_widths)
    all_layer_widths = [input_dim] + hidden_layer_widths + [output_dim]
    final_linear_layer = torch.nn.Linear(hidden_layer_widths[-1], output_dim, bias=include_bias)
    layer_list = []
    for i in range(num_layers):
        w0 = all_layer_widths[i]
        w1 = all_layer_widths[i + 1]
        layer_list += [torch.nn.Linear(w0, w1, bias=include_bias), torch.nn.ReLU()]
    layer_list += [final_linear_layer]
    return layer_list


def get_acas_model(input_width: int,
                   output_width: int,
                   cfg_model: DictConfig) -> torch.nn.Module:
    hidden_layer_widths = [cfg_model.neurons_per_layer] * cfg_model.num_relu_layers
    layer_list = build_relu_layers(input_width,
                                   hidden_layer_widths,
                                   output_width,
                                   cfg_model.include_bias)
    model = torch.nn.Sequential(*layer_list)
    return model
