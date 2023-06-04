# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Build all the components of the welcome tab, is called by the main_app.py file."""
import omegaconf
import torch
from bokeh.layouts import column
from bokeh.models import Div, Markup, Paragraph, PreText


def get_model_desc(network: torch.nn.Module, state_dict_fullfilename: str) -> Div:
    """Build the model description."""
    network_str = str(network).replace("\n", "<br />")
    styles_dict = {}
    div_text = f"""
        <h3>Model</h3><br />
        <code>{network_str}</code><br />
        <br />

        <strong>Total number of parameters: {sum(p.numel() for p in network.parameters())}
        </strong>"""

    # https://stackoverflow.com/questions/38979784/how-to-automatically-adjust-bokeh-plot-size-to-the-screen-size-of-device-used
    architecture_desc = Div(text=div_text, width=500, height=400, styles=styles_dict)
    fullfilename_text = f"Model loaded from: {state_dict_fullfilename}"
    fullfilename_desc = PreText(text=fullfilename_text, width=1000, height=20)
    model_desc = column(architecture_desc, fullfilename_desc)
    return model_desc


def build_metadata_body(cfg_data: omegaconf.DictConfig) -> Markup:
    """Build the metadata body."""
    output_str_lines = (
        ["Output dimensions"]
        + [f"{idx}: {oname}" for idx, oname in enumerate(cfg_data.output.names)]
        + ["-" * 80]
    )
    input_str_lines = (
        ["Input dimensions"]
        + [
            f"{idx}: {t[0]} {t[1]} {t[2]}"
            for idx, t in enumerate(
                zip(cfg_data.input.names, cfg_data.input.lower, cfg_data.input.upper)
            )
        ]
        + ["-" * 80]
    )
    metadata_text = "\n".join(input_str_lines + output_str_lines)
    metadata_body = PreText(text=metadata_text, width=1000, height=100)
    return metadata_body


def welcome_layout(cfg_data, model, state_dict_fullfilename):
    """Build the welcome layout."""
    welcome_title = Paragraph(
        text="Neural Network Visualizer by Kyle MATOBA and Arnaud PANNATIER",
        width=1000,
        height=20,
    )

    welcome_body = get_model_desc(model, state_dict_fullfilename)

    metadata_body = build_metadata_body(cfg_data)

    return column(welcome_title, welcome_body, metadata_body)
