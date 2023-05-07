import os
import sys
from pathlib import Path

import hydra

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(sys.path)


import torch
import torch.optim
import torch.utils.data
from bokeh.models import Div
from bokeh.plotting import save
from omegaconf import DictConfig


@torch.no_grad()
@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    pra = 0
    tau = 0

    model_ident_pattern = cfg.model.ident_pattern

    filename = model_ident_pattern.format(
        pra, tau, cfg.model.num_relu_layers, cfg.model.neurons_per_layer
    )
    state_dict_fullfilename = os.path.join(cfg.paths.nnet, filename + ".pt")

    saved = torch.load(state_dict_fullfilename, map_location="cpu")
    model = saved["model"]

    model_str = str(model).replace("\n", "<br />")

    div = Div(text=f"""
    <h3>Model</h3><br />
    <code>{model_str}</code><br />
    <br />
    
    <strong>Total number of parameters: {sum(p.numel() for p in model.parameters())}
    </strong>""", width=500, height=400, styles={"background-color": "#eeeeee", "border": "1px solid black", "padding-left": "30px"})
    save(div)


if __name__ == "__main__":
    main()
