import math

import bokeh.palettes
import numpy as np
import torch
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    Dropdown,
    Paragraph,
    Range1d,
    Slider,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

# https://docs.bokeh.org/en/latest/docs/reference/models/widgets/buttons.html#bokeh.models.Dropdown
# https://stackoverflow.com/questions/60715996/bokeh-2-0-dropdown-missing-value-attribute

## Add dev for some live reloading
# bokeh serve --show streaming/bokeh_slider_example.py --dev
# https://stackoverflow.com/questions/51144743/make-the-colour-and-marker-of-bokeh-plot-scatter-points-dependent-on-dataframe-v
# https://stackoverflow.com/questions/50013378/how-to-draw-a-circle-plot-the-linearcolormapper-using-python-bokeh/50048081#50048081


dim_in = 3
dim_out = 5
network = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), torch.nn.ReLU())


def dropdown_handler(event):
    palette_name = event.item
    print(f"{palette_name=}")
    palette_paragraph.text = event.item

    palette_class = bokeh.palettes.all_palettes[palette_name]
    assert (
        dim_out in palette_class.keys()
    ), f"palette_name = {palette_name} does not admit this many classes"
    palette = palette_class[dim_out]
    # I don't have an explanation of why the other cases don't work
    # But this works
    scatter.glyph.fill_color = factor_cmap('i', palette, outputs_names)

def build_dropdown() -> Dropdown:
    menu = [(_, _) for _ in bokeh.palettes.all_palettes.keys()]
    dropdown = Dropdown(label="Bokeh palette", button_type="warning", menu=menu)
    dropdown.on_click(dropdown_handler)
    return dropdown


def my_text_input_handler(attr, old, new):
    msg = "Neural network filename: {0}".format(new)
    text_output.text = msg


def slider_callback(attr, old, new):
    value = slider.value
    pointset[:, 0] = value
    y_out = network(torch.tensor(pointset, dtype=dtype)).detach()
    i_out = [outputs_names[i%dim_out] for i in range(num_points)]
    source.data = {"x": np.array(pointset), "y": np.array(y_out), "i": i_out}


def persp_proj(x_in: np.ndarray) -> np.ndarray:
    n = x_in.shape[1]
    p = 1.0
    x_out = x_in[:, :-1] * p * n / (p * n - x_in[:, -1].reshape(-1, 1))
    return x_out


"""
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
    test4 = projection_mat @ rotation_x @ rotation_y @ rotation_z @ points.T
    test5 = test4
    projected_points = test5.T
"""

# input metadata

num_points = 40
if dim_in == 5:
    num_slider_values = 100
    inputs_lower = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    inputs_upper = np.array([[100.0, 100.0, 2 * math.pi, 2 * math.pi, 1.0]])
    inputs_names = ["speed_x", "speed_y", "angle1", "angle2", "scale"]
    outputs_names = [
        "strong_left",
        "weak_left",
        "straight",
        "weak_right",
        "strong_right",
    ]
elif dim_in == 3:
    num_slider_values = 100
    inputs_lower = np.array([[0.0, 0.0, 0.0]])
    inputs_upper = np.array([[100.0, 100.0, 2 * math.pi]])
    inputs_names = ["speed_x", "speed_y", "angle1"]
    outputs_names = [
        "strong_left",
        "weak_left",
        "straight",
        "weak_right",
        "strong_right",
    ]
else:
    raise ValueError(f"dim_in = {dim_in} not configured")


# pointset = inputs_lower + torch.rand(num_points, dim_in) @ inputs_upper.T
pointset = inputs_lower + np.random.rand(num_points, dim_in) @ inputs_upper.T

# Choose projection values, or perspective projection?

# logits = network(pointset)
# preds = logits.argmax(1)

dtype = torch.float32

y = np.array(network(torch.tensor(pointset, dtype=dtype)).detach())
source = ColumnDataSource(
    data={"x": pointset, "y": y, "i": [outputs_names[i] for i in y.argmax(1)]}
)

proj_dim = 0
rem_dim1 = 1
rem_dim2 = 2

min_slider_val = inputs_lower[0, proj_dim]
max_slider_val = inputs_upper[0, proj_dim]
step_slider_val = (max_slider_val - min_slider_val) / num_slider_values

slider = Slider(
    start=min_slider_val,
    end=max_slider_val,
    value=(min_slider_val + max_slider_val) / 2,
    step=step_slider_val,
    title=inputs_names[proj_dim],
)

slider.on_change("value", slider_callback)
default_palette_name = "Spectral"

text_output = Paragraph(text=f"", width=1000, height=100)
palette_paragraph = Paragraph(text=default_palette_name, width=1000, height=100)

text_input = TextInput(value="Enter Neural network filename", title="Label: ")
text_input.on_change("value", my_text_input_handler)

palette_name = palette_paragraph.text
palette_name = "PRGn"
palette_class = bokeh.palettes.all_palettes[palette_name]
assert (
    dim_out in palette_class.keys()
), f"palette_name = {palette_name} does not admit this many classes"
print(dim_out)
palette = palette_class[dim_out]
print(palette)


plot = figure()
scatter = plot.scatter(
    source=source,
    x="x",
    y="y",
    size=10,
    fill_color=factor_cmap("i", palette, outputs_names),
)


palette_name = "YlGnBu"
palette_class = bokeh.palettes.all_palettes[palette_name]
assert (
    dim_out in palette_class.keys()
), f"palette_name = {palette_name} does not admit this many classes"
palette = palette_class[dim_out]
print(palette)
scatter.glyph.fill_color.transform.update(palette=palette)

print("ID scatter", id(scatter.glyph.fill_color.transform.palette))
print(scatter)

plot.y_range = Range1d(0, 100)
plot.x_range = Range1d(0, 100)
plot.xaxis.axis_label = inputs_names[rem_dim1]
plot.yaxis.axis_label = inputs_names[rem_dim2]

doc = curdoc()

layout = column(slider, plot)
doc.add_root(layout)

dropdown = build_dropdown()
layout = column(text_input, text_output, dropdown, palette_paragraph)
doc.add_root(layout)
doc.title = "Neural Network Introspector"
