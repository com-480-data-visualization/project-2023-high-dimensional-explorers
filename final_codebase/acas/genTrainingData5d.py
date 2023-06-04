# python3 genTrainingData5d.py
import math
import os
import logging
import time

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(level=logging.DEBUG)
logging.Formatter.converter = time.gmtime


@hydra.main(version_base=None,
            config_path="../config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    cfg_data = cfg.hcas.data
    ident = cfg_data.experiment_ident
    training_data_file_pattern = cfg_data.training_data_pattern

    if cfg.paths.tilde:
        tilde = cfg.paths.tilde
    else:
        tilde = os.path.expanduser("~")
    training_data_dir = os.path.join(tilde, cfg.paths.training_data)
    qtable_dir = os.path.join(tilde, cfg.paths.qtable)
    os.makedirs(training_data_dir, exist_ok=True)
    vowns = cfg_data.vowns
    vints = cfg_data.vints

    thetas = np.linspace(-math.pi, +math.pi, cfg_data.n_theta).tolist()
    psis = np.linspace(-math.pi, +math.pi, cfg_data.n_psis).tolist()
    ranges = cfg_data.ranges

    x_raw = np.array(
        [
            [r * np.cos(t), r * np.sin(t), p, vo, vi]
            for vi in vints
            for vo in vowns
            for p in psis
            for t in thetas
            for r in ranges
        ]
    )
    os.makedirs(qtable_dir, exist_ok=True)
    filename = "baseline.h5"
    fullfilename = os.path.join(qtable_dir, filename)
    logger.info(f"Data format is: x, y, psi, v_own, v_intruder")
    f = h5py.File(fullfilename, "r")
    q_mat = np.array(f["q"])
    f.close()
    q_mat = q_mat.T

    qq_mat = (q_mat - q_mat.mean()) / (q_mat.max() - q_mat.min())

    x_mean = x_raw.mean(0)
    x_maxs = x_raw.max(0)
    x_mins = x_raw.min(0)
    x_rng = x_maxs - x_mins

    x_rng0 = np.where(x_rng == 0.0, 1.0, x_rng)
    x = (x_raw - x_mean) / x_rng0

    acts = list(range(5))
    ns2 = len(ranges) * len(thetas) * len(psis) * len(vowns) * len(vints) * len(acts)
    ns3 = len(ranges) * len(thetas) * len(psis) * len(vowns) * len(vints)

    for tau in [0, 5, 10, 15, 20, 30, 40, 60]:
        # tau = 0
        Qsub = qq_mat[tau * ns2: (tau + 1) * ns2]
        for pra in range(5):
            # pra = 0
            Qsubsub = Qsub[pra * ns3: (pra + 1) * ns3]
            training_data_filename = training_data_file_pattern.format(ident, pra, tau)
            training_data_fullfilename = os.path.join(
                training_data_dir, training_data_filename
            )
            # filename = training_data_fullfile_pattern.format(pra, tau)
            logger.info(
                "Saving {} values to {}".format(
                    Qsubsub.shape[0], training_data_fullfilename
                )
            )
            with h5py.File(training_data_fullfilename, "w") as h:
                h.create_dataset("x", data=x)
                h.create_dataset("y", data=Qsubsub)
                h.create_dataset("min_inputs", data=x_maxs)
                h.create_dataset("max_inputs", data=x_mins)


if __name__ == "__main__":
    main()
