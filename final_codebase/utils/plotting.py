import os

import matplotlib
import matplotlib.pyplot as plt


def initialise_pgf_plots(texsystem: str, font_family: str) -> None:
    """
    font_family = "serif"
    utils.plotting.initialise_pgf_plots("pdflatex", font_family)
    """
    plt.switch_backend("pgf")
    # https://matplotlib.org/users/customizing.html
    pgf_with_rc_fonts = {
        "pgf.texsystem": texsystem,
        "font.family": font_family,
        "font.serif": [],
        "text.usetex": True,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)


def smart_save_fig(
    fig: matplotlib.figure.Figure, ident: str, fig_format: str, filepath: str
) -> str:
    filename = "{}.{}".format(ident, fig_format)
    os.makedirs(filepath, exist_ok=True)
    fig_path = os.path.join(filepath, filename)
    fig.savefig(fig_path)
    return fig_path
