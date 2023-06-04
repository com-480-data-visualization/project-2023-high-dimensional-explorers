# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Build all the components of the rotation tab, is called by the main_app.py file."""
import math
import time
from functools import partial

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Paragraph, RadioButtonGroup, Toggle
from bokeh.plotting import figure
from data_generation import build_pointset
from projections import (
    all_angular_proj,
    get_proj32t_mat,
    persp_proj_down_to_dim_from_end,
    rotation_matrix_prod,
)


def _plot_at_angle(plot, source, angle, points, cfg_data, rotation_plot_dims, proj_type="all_angular", points3=None):
    """Plot the points at a given angle.
    Callback function which is called repeatedly by the bokeh server.
    """
    if "perspective" == proj_type:
        post_mult = rotation_matrix_prod(angle) @ get_proj32t_mat()
        projected_points = points3 @ post_mult
    elif "all_angular" == proj_type:
        projected_points = all_angular_proj(points, angle, 2)
    else:
        raise ValueError(f"do not know about proj_type = {proj_type}")

    k1 = cfg_data.input.names[rotation_plot_dims[0]]
    k2 = cfg_data.input.names[rotation_plot_dims[1]]
    # f.scatter(source=source, x=", y=
    #
    source.data[f"p_{k1}"] = projected_points[:, 0]
    source.data[f"p_{k2}"] = projected_points[:, 1]

    angle_in_deg = math.fmod(angle * 180 / math.pi, 360)
    plot.title.text = f"angle: {angle_in_deg:.2f} [°] "


def angle_callback(toggle, plot, source, points, cfg_data, rotation_plot_dims, proj_type, points3):
    """Call the plot_at_angle function if toggle is active.
    Callback function for the toggle button.

    Starts and stops the rotation.
    Angle is computed from the time.
    """
    if toggle.active:
        angle = time.time()
        _plot_at_angle(plot, source, angle, points, cfg_data, rotation_plot_dims, proj_type, points3)


def rotation_layout(cfg_data, cfg_viz, source, cmap, rotation_plot_dims, rotation_other_dims):
    """Build the layout for the rotation tab."""
    names = cfg_data.input.names
    toggle = Toggle(label="Start / stop", button_type="success")

    projection_names = ["all_angular", "perspective"]
    radio = RadioButtonGroup(
        labels=projection_names, active=projection_names.index(cfg_viz.proj_type)
    )
    points = build_pointset(
        cfg_viz.num_points,
        cfg_viz.pointset_type,
        cfg_data.input.dim,
        np.array(cfg_data.input.lower),
        np.array(cfg_data.input.upper),
    )
    points_reordered = np.concatenate((points[:, rotation_plot_dims], points[:, rotation_other_dims]), 1)

    points3 = persp_proj_down_to_dim_from_end(points_reordered, 3)

    TOOLTIPS = [(_, "@" + _) for _ in cfg_data.input.names] + [("label", "@label")]

    f = figure(
        height=400,
        width=400,
        toolbar_location=None,
        tools="lasso_select",
        active_drag="lasso_select",
        tooltips=TOOLTIPS,
        sizing_mode="scale_height",
        match_aspect=True,
    )

    k1 = cfg_data.input.names[rotation_plot_dims[0]]
    k2 = cfg_data.input.names[rotation_plot_dims[1]]
    f.scatter(source=source, x=f"p_{k1}", y=f"p_{k2}", fill_color=cmap, size=cfg_viz.marker_size)
    f.xgrid.grid_line_color = None
    f.xaxis.axis_label = f"p_{k1}"
    f.yaxis.axis_label = f"p_{k2}"

    body = column(f, row(toggle, radio), sizing_mode="scale_height")
    title = Paragraph(text="Rotation", width=1000, height=20)

    angle_cb = partial(
        angle_callback, toggle, f, source, points, cfg_data, rotation_plot_dims, cfg_viz.proj_type, points3
    )

    def radio_callback(idx):
        return partial(
            angle_callback,
            toggle,
            f,
            source,
            points,
            cfg_data,
            rotation_plot_dims,
            projection_names[idx],
            points3,
        )

    return (
        column(title, body, sizing_mode="scale_height"),
        angle_cb,
        radio,
        radio_callback,
    )
