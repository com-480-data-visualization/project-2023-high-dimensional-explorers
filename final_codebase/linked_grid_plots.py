# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Build all the components of the plot tab, is called by the main_app.py file."""
from functools import partial

import numpy as np
from bokeh.layouts import column, layout, row
from bokeh.models import Button, CustomJS, NumeralTickFormatter, Slider
from bokeh.plotting import figure


def create_figure(
    x,
    y,
    sz,
    source,
    cmap,
    has_xlabel,
    has_ylabel,
    ranges,
    tooltips,
    pad_label_length=10,
):
    """Create a figure with shared source and ranges."""

    ranges_dict = {}
    if ranges[x] is not None:
        ranges_dict["x_range"] = ranges[x]

    if ranges[y] is not None:
        ranges_dict["y_range"] = ranges[y]

    f = figure(
        width=220,
        height=220,
        toolbar_location=None,
        tools="lasso_select",
        active_drag="lasso_select",
        tooltips=tooltips,
        min_border_left=50,
        min_border_bottom=50,
    )
    f.scatter(source=source, x=x, y=y, fill_color=cmap, size=sz)

    xlabel = x if has_xlabel else " "
    ylabel = y if has_ylabel else " "
    f.xaxis.axis_label = f"{xlabel:<{pad_label_length}}"
    f.yaxis.axis_label = f"{ylabel:<{pad_label_length}}"
    f.xaxis.formatter = NumeralTickFormatter(format="0.0")
    f.yaxis.formatter = NumeralTickFormatter(format="0.0")
    if ranges[x] is None:
        ranges[x] = f.x_range
    if ranges[y] is None:
        ranges[y] = f.y_range

    return f


def update_source(attr, old, new, name, forward, source):
    """Update the source.

    Collapse all the points of the source at dimension [name] to the value [new] given by the slider.
    """
    d = source.data
    d[name] = np.full_like(d[name], new)
    source.data["label"] = forward(d)


def reset_source(name, forward, source, original_points):
    """Reset the source to the uncollapsed points."""
    d = source.data
    d[name] = original_points[name]
    source.data["label"] = forward(d)


def hide_collapsed(attr, old, new, plots, name):
    """Hide the plots where some dimension collapsed.

    Callback function for the sliders.
    Can be changed to remove the plot instead of hiding it.
    p.visible = False -> remove the plot completely, messes with the layout
    p.renderers = [] -> remove the plot, white square remains
    The solution we choose keep empty axes.
    """
    for p in plots[name]:
        for renderer in p.renderers:
            renderer.visible = False


def show_reset(plots, name):
    """Show again the plots that were hidden by the collapse."""
    for p in plots[name]:
        for renderer in p.renderers:
            renderer.visible = True


def tab_layout(cfg_data, sz, source, cmap, forward, sample):
    """Build the layout of the tab."""
    c = []
    names = cfg_data.input.names
    ranges = {name: None for name in names}
    plots = {name: [] for name in names}

    original_points = source.data.copy()

    TOOLTIPS = [(_, "@" + _) for _ in cfg_data.input.names] + [("label", "@label")]

    for i, y in enumerate(names[1:], 1):
        r = []
        for x in names[:i]:
            f = create_figure(
                x, y, sz, source, cmap, y == names[-1], x == names[0], ranges, TOOLTIPS
            )
            r.append(f)
            plots[x].append(f)
            plots[y].append(f)
        c.append(r)

    sliders = [
        Slider(start=-0.5, end=0.5, value=0, step=0.01, title=name) for name in names
    ]
    reset = [Button(label="Reset", button_type="success") for _ in names]

    for name, slider in zip(names, sliders):
        update_source_callback = partial(
            update_source, name=name, forward=forward, source=source
        )
        slider.on_change("value", update_source_callback)

        hide_collapsed_callback = partial(hide_collapsed, plots=plots, name=name)
        slider.on_change("value", hide_collapsed_callback)

    sliders_rows = []
    for slider, button in zip(sliders, reset):
        # this should be js as otherwise the slider does not move
        reset_callback = CustomJS(args={"slider": slider}, code="""slider.value=0;""")
        button.js_on_click(reset_callback)

        reset_source_callback = partial(
            reset_source,
            name=slider.title,
            forward=forward,
            source=source,
            original_points=original_points,
        )
        button.on_click(reset_source_callback)

        show_reset_callback = partial(show_reset, plots=plots, name=slider.title)
        button.on_click(show_reset_callback)

        sliders_rows.append(row(slider, button))

    resample_button = Button(label="Resample", button_type="success")

    def resample(source=source):
        s = sample()
        source.data = dict(s.data)

    resample_button.on_click(resample)
    sliders_rows.append(resample_button)
    return row([layout(c), column(sliders_rows)])
