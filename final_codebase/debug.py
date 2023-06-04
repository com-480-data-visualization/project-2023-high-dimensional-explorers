# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Build all the components of the debug tab, is called by the main_app.py file."""
import datetime as dt
import getpass
import pprint
import socket

import omegaconf
from bokeh.layouts import column
from bokeh.models import Markup, Paragraph, PreText
from hydra.core.hydra_config import HydraConfig

import utils.git


def build_debug_body(cfg: omegaconf.DictConfig, log_fullfilename: str) -> Markup:
    """Build the debug body."""
    iwtc = utils.git.is_working_tree_clean()

    username = getpass.getuser()
    hostname = socket.gethostname()

    git_describe = utils.git.get_git_describe()
    timestr = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f %Z")
    footer = "-" * 80 + "\n" + f"{username} @ {hostname} -> {timestr}"

    # https://hydra.cc/docs/configure_hydra/intro/#hydraruntime
    hconfig = HydraConfig.get()

    hydra_info = pprint.pformat(dict(hconfig))
    git_info = f"git has = {git_describe}, work tree clean = {iwtc}"
    log_info = f"Python logs at {log_fullfilename}"
    metadata_text = pprint.pformat(dict(cfg))

    debug_body_lines = [git_info, log_info, metadata_text, hydra_info, footer]
    debug_body_text = "\n".join(debug_body_lines)
    debug_body = PreText(
        text=debug_body_text, width=1000, height=800, styles={"overflow-x": "scroll"}
    )
    return debug_body


def debug_layout(cfg: omegaconf.DictConfig, log_fullfilename: str):
    """Build the debug layout."""
    debug_title = Paragraph(
        text=f"This debug panel contains metadata about the run to help diagnose development problems",
        width=1000,
        height=20,
    )
    debug_body = build_debug_body(cfg, log_fullfilename)

    return column(debug_title, debug_body)
