# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Main application file.

Load the configuration file and start the bokeh server.
"""

import functools
import math
import os
from typing import Tuple

import bokeh.palettes
import data_generation
import hydra
import numpy as np
import torch
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models import ColumnDataSource, TabPanel, Tabs
from bokeh.server.server import Server
from bokeh.transform import factor_cmap
from tornado.ioloop import IOLoop
from omegaconf import DictConfig, OmegaConf

from debug import debug_layout
import linked_grid_plots
import utils.config
from misc import add_filehandler_to_root_logger, get_logger
from omegaconf import DictConfig, OmegaConf
from rotation_plot import rotation_layout
from welcome import welcome_layout

# kill -9  `lsof -i:5006 -t`

logger = get_logger()
OmegaConf.register_new_resolver("math", lambda attr: getattr(math, attr))


def get_palette(palette_name: str, dim_out: int):
    """Get a palette from bokeh.palettes."""
    palette_class = bokeh.palettes.all_palettes[palette_name]
    if False:
        palette_name = "HighContrast"
        palette_class = bokeh.palettes.all_palettes[palette_name]
        palette = palette_class[3]

    if 2 == dim_out:
        if dim_out in palette_class.keys():
            palette_out = palette_class[dim_out]
        else:
            # palette_name = "HighContrast"
            # palette_class = bokeh.palettes.all_palettes[palette_name]
            # palette = palette_class[3]
            # # palette_out = (palette[0], palette[2])
            # palette_out = (palette[0], palette[1])

            palette = palette_class[3]
            p = palette_class[3]
            palette_out = (p[0], p[2])
    else:
        assert (
            dim_out in palette_class.keys()
        ), f"palette_name = {palette_name} does not admit this many classes"
        palette_out = palette_class[dim_out]
    return palette_out


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Load the configuration file and start the bokeh server.

    the main function is wrapped for the bokeh server.
    it should have the signature f(doc: bokeh.document.Document)
    """
    print("Opening Bokeh application.")
    f_wrapped = functools.partial(f, cfg=cfg)
    bokeh_app = Application(FunctionHandler(f_wrapped))
    io_loop = IOLoop.current()
    server = Server({"/": bokeh_app}, io_loop=io_loop)
    server.start()
    io_loop.add_callback(server.show, "/")
    io_loop.start()


def load_hcas_network(tilde: str, cfg_hcas: str, cfg_paths: str) -> Tuple[torch.nn.Module, str]:
    model_ident_pattern = cfg_hcas.model.ident_pattern
    filename = model_ident_pattern.format(
        cfg_hcas.pra, cfg_hcas.tau, cfg_hcas.model.num_relu_layers, cfg_hcas.model.neurons_per_layer
    )
    filedir = os.path.join(tilde, cfg_paths.nnet)
    saved_fullfilename = os.path.join(filedir, filename + ".pt")

    saved = torch.load(saved_fullfilename, map_location="cpu")
    network = saved["model"]
    return network, saved_fullfilename


def load_cartpole_network(tilde: str, cfg_cartpole: str, cfg_paths: str):
    cfg_model = cfg_cartpole.model
    model_ident_pattern = cfg_model.ident_pattern

    filename = model_ident_pattern.format(cfg_model.num_relu_layers,
                                          cfg_model.neurons_per_layer)

    filedir = os.path.join(tilde, cfg_paths.nnet)
    saved_fullfilename = os.path.join(filedir, filename + ".pt")

    saved = torch.load(saved_fullfilename, map_location="cpu")
    network = saved["model"]
    return network, saved_fullfilename


def f(doc: bokeh.document.Document,
      cfg: DictConfig):
    """Wrap the main function to be called by the bokeh server.

    import the tab from the other files and add them to the document.
    initialize the data and the callbacks.
    """
    device, dtype = utils.config.setup(cfg)

    if cfg.paths.tilde:
        tilde = cfg.paths.tilde
    else:
        tilde = os.path.expanduser("~")

    if "hcas" == cfg.usecase:
        network, saved_fullfilename = load_hcas_network(tilde, cfg.hcas, cfg.paths)
        rotation_plot_dims = cfg.hcas.rotation_plot_dims
        rotation_other_dims = cfg.hcas.rotation_other_dims
    elif "cartpole" == cfg.usecase:
        network, saved_fullfilename = load_cartpole_network(tilde, cfg.cartpole, cfg.paths)
        rotation_plot_dims = cfg.cartpole.rotation_plot_dims
        rotation_other_dims = cfg.cartpole.rotation_other_dims
    else:
        raise ValueError(f"Do not know about use case = {cfg.usecase}")
    log_dir = os.path.join(tilde, cfg.paths.logs)

    log_fullfilename = add_filehandler_to_root_logger(log_dir)
    logger.info(f"Added filehandler at {log_fullfilename}")
    cfg_data = cfg[cfg.usecase].data

    palette = get_palette(cfg.viz.palette_name, cfg_data.output.dim)
    cmap = factor_cmap("label", palette, np.array(cfg_data.output.names))

    def forward(d):
        to_stack = [d[k] for k in cfg_data.input.names]
        pointset = np.stack(to_stack, 1)
        model_input = torch.tensor(pointset, dtype=dtype)
        logits = network(model_input).detach()
        output_names = np.array(cfg_data.output.names)
        return [output_names[m] for m in logits.argmax(1)]

    def sample():
        pointset = data_generation.build_pointset(
            cfg.viz.num_points,
            cfg.viz.pointset_type,
            cfg_data.input.dim,
            np.array(cfg_data.input.lower),
            np.array(cfg_data.input.upper),
        )
        d = {name: pointset[:, idx] for idx, name in enumerate(cfg_data.input.names)}
        d["label"] = forward(d)

        k1 = cfg_data.input.names[rotation_plot_dims[0]]
        k2 = cfg_data.input.names[rotation_plot_dims[1]]
        d[f"p_{k1}"] = d[k1]
        d[f"p_{k2}"] = d[k2]

        source = ColumnDataSource(data=d)
        return source

    source = sample()
    tab = linked_grid_plots.tab_layout(cfg_data, cfg.viz.marker_size, source, cmap, forward, sample)

    (
        rotation,
        rotation_callback_fun,
        rotation_radio,
        rotation_radio_callback_fn,
    ) = rotation_layout(cfg_data, cfg.viz, source, cmap, rotation_plot_dims, rotation_other_dims)
    rotation_tab = TabPanel(child=rotation, title="Rotation")

    plots_tab = TabPanel(child=tab, title="Plots")

    debug_tab = debug_layout(cfg, log_fullfilename)
    debug_tab = TabPanel(child=debug_tab, title="Debug")

    welcome_tab = welcome_layout(cfg_data, network, saved_fullfilename)
    welcome_tab = TabPanel(child=welcome_tab, title="Start Here")

    tabs = [welcome_tab, plots_tab, rotation_tab, debug_tab]
    layout = Tabs(tabs=tabs)
    doc.add_root(layout)

    callback_ids = {}

    callback_ids["rotation"] = doc.add_periodic_callback(
        rotation_callback_fun, cfg.viz.callback_period_milliseconds
    )

    def replace_periodic_callback(attr, old, new, callback_ids=callback_ids):
        cb = rotation_radio_callback_fn(new)
        doc.remove_periodic_callback(callback_ids["rotation"])
        callback_ids["rotation"] = doc.add_periodic_callback(
            cb, cfg.viz.callback_period_milliseconds
        )

    rotation_radio.on_change("active", replace_periodic_callback)
    doc.title = cfg.viz.page_title


if __name__ == "__main__":
    main()
