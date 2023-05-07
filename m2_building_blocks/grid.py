import os
import sys
from pathlib import Path

import hydra

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations

import config
import numpy as np
import torch
import torch.optim
import torch.utils.data
from bokeh.layouts import gridplot
from bokeh.palettes import Set1_9
from bokeh.plotting import figure, save
from omegaconf import DictConfig


@torch.no_grad()
@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    _, dtype = config.setup(cfg)

    pra = 0
    tau = 0

    model_ident_pattern = cfg.model.ident_pattern

    filename = model_ident_pattern.format(
        pra, tau, cfg.model.num_relu_layers, cfg.model.neurons_per_layer
    )
    state_dict_fullfilename = os.path.join(cfg.paths.nnet, filename + ".pt")

    p = Path(state_dict_fullfilename)

    saved = torch.load(state_dict_fullfilename, map_location="cpu")
    model = saved["model"]
    training_data_file_pattern = cfg.data.training_data_pattern
    ident = cfg.hcas.experiment_ident

    print("(tau, pra) = ({}, {})".format(tau, pra))
    training_data_filename = training_data_file_pattern.format(ident, pra, tau)


    num_points = 1024
    x = torch.rand((num_points, 5))

    o = model(x).argmax(1).numpy().astype(np.uint8)
    print(o.shape)

    o = [Set1_9[i] for i in o]

    x = x.numpy()

    ranges = [None] * 5

    figs = []
    for i, j in combinations(range(5), 2):
        print(i, j)

        ranges_dict = {}
        if ranges[i] is not None:
            ranges_dict["x_range"] = ranges[i]

        if ranges[j] is not None:
            ranges_dict["y_range"] = ranges[j]

        fig = figure(
            width=300,
            height=300,
            title=f"{i} vs {j}",
            **ranges_dict,
        )
        fig.circle(x[:, i], x[:, j], color=o, size=10, line_color=None)
        figs.append(fig)

        if ranges[i] is None:
            ranges[i] = fig.x_range
        if ranges[j] is None:
            ranges[j] = fig.y_range

    grid = gridplot(figs, ncols=5)

    save(grid)


if __name__ == "__main__":
    main()
