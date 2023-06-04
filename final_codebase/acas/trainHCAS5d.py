import os
import time
import logging
from typing import Tuple

# warnings.filterwarnings("ignore", category=Warning)
# warnings.filterwarnings("error")
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import h5py

import utils.config
import acas.networks

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(level=logging.DEBUG)
logging.Formatter.converter = time.gmtime


# https://stackoverflow.com/questions/45667439/how-to-type-annotate-tensorflow-session-in-python3

# From: https://arxiv.org/pdf/1912.07084.pdf
# The networks have three inputs, one for each remaining
# state variable. Each network has five hidden layers of
# length 25 and nine outputs, one for each advisory. Nine
# smaller networks are trained instead of a single large
# network to reduce run-time required to evaluate each network.
# Each network was trained for 3000 epochs in 30 minutes using
# Tensorflow, resulting in nine neural networks that reduce
# required storage from 1.22GB to 103KB while maintaining the
# correct advisory 94.9% of the time.


def _cust_acc_pt(y_true, y_pred):
    maxes_pred = y_pred.argmax(axis=1)
    inds = y_true.argmax(axis=1)
    diff = (inds - maxes_pred).abs()
    ones = torch.ones_like(diff, dtype=torch.float32)
    zeros = torch.zeros_like(diff, dtype=torch.float32)
    l = torch.where(diff < 0.5, ones, zeros)
    out = torch.mean(l)
    return out


def _fit_model_kernel(x: torch.Tensor,
                      y: torch.Tensor,
                      model: torch.nn.Module,
                      cfg_fitting: DictConfig,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    log_every_epoch = 5
    num_rows, input_width = x.shape
    optim = torch.optim.Adam
    optim_kwargs = {"lr": cfg_fitting.lr,
                    "betas": (0.9, 0.999)}

    criterion = torch.nn.CrossEntropyLoss()
    train_dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg_fitting.batch_size)
    optimizer = optim(model.parameters(), **optim_kwargs)

    losses = torch.empty((cfg_fitting.num_epochs,))
    accuracies = torch.empty((cfg_fitting.num_epochs,))
    probs = torch.ones((num_rows,))
    for epoch_idx in range(cfg_fitting.num_epochs):
        for batch_idx, (x_, y_) in enumerate(dataloader):
            x_ = x_.to(device)
            y_ = y_.to(device)

            y_pred = model.forward(x_)
            loss = criterion(y_pred, y_.argmax(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_inds = torch.multinomial(probs,
                                      cfg_fitting.test_size, replacement=False)
        x_test = x[test_inds, :].to(device)
        yhat = model(x_test)
        cust_acc = _cust_acc_pt(y[test_inds].to(device), yhat)
        accuracies[epoch_idx] = cust_acc
        losses[epoch_idx] = loss.item()
        if (epoch_idx > 0) and epoch_idx % log_every_epoch == 0:
            logger.info("Epoch {:4d} / {:4d} -- acc, loss = {:.4f}, {:.4f}".format(epoch_idx,
                                                                             cfg_fitting.num_epochs,
                                                                             accuracies[epoch_idx],
                                                                             losses[epoch_idx]))
    return accuracies, losses, model


def subset_by_psi(x: torch.Tensor,
                  y: torch.Tensor,
                  min_scaled_psi: float,
                  max_scaled_psi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    psi_ind = 2
    is_above_min = x[:, psi_ind] >= min_scaled_psi
    is_below_max = x[:, psi_ind] <= max_scaled_psi
    subset_rows = torch.logical_and(is_above_min, is_below_max)

    x_train = x[subset_rows, :]
    y_train = y[subset_rows, :]
    return x_train, y_train


@hydra.main(version_base=None,
            config_path="../config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device, dtype = utils.config.setup(cfg)
    taus = [0]
    pras = [0]

    num_taus = len(taus)
    num_pras = len(pras)

    cfg_model = cfg.hcas.model
    cfg_data = cfg.hcas.data
    cfg_train = cfg.hcas.train

    model_ident_pattern = cfg_model.ident_pattern
    training_data_file_pattern = cfg_data.training_data_pattern
    ident = cfg_data.experiment_ident

    if cfg.paths.tilde:
        tilde = cfg.paths.tilde
    else:
        tilde = os.path.expanduser("~")
    in_filedir = os.path.join(tilde, cfg.paths.training_data)
    out_filedir = os.path.join(tilde, cfg.paths.nnet)
    os.makedirs(out_filedir, exist_ok=True)

    accuracies_dict = dict()
    for t_idx, tau in enumerate(taus):
        for p_idx, pra in enumerate(pras):
            logger.info("(tau, pra) = ({}, {})".format(tau, pra))
            training_data_filename = training_data_file_pattern.format(ident, pra, tau)
            f_fullfilename = os.path.join(in_filedir, training_data_filename)

            f = h5py.File(f_fullfilename, "r")
            x = torch.tensor(np.array(f["x"]), dtype=dtype)
            y = torch.tensor(np.array(f["y"]), dtype=dtype)
            if False:
                fig, axs = plt.subplots(1, 1, squeeze=False)
                keeprows = (np.abs(x[:, 2]) < 1e-15) & \
                           (np.abs(x[:, 3]) < 1e-15) & \
                           (np.abs(x[:, 4]) < 1e-15)

                xkeep = x[keeprows, :]
                ykeep = y[keeprows, :]
                axs[0, 0].scatter(xkeep[:, 0], xkeep[:, 1])

            x_train, y_train = subset_by_psi(x, y, cfg_data.min_scaled_psi, cfg_data.max_scaled_psi)

            output_width = y.shape[1]
            num_rows, input_width = x.shape
            model = acas.networks.get_acas_model(input_width, output_width, cfg_model).to(device)

            logger.info("Fitting DNN with {} rows".format(x_train.shape[0]))
            accuracies, losses, model = _fit_model_kernel(x_train,
                                                          y_train,
                                                          model,
                                                          cfg_train,
                                                          device)
            accuracies_dict[(tau, pra)] = accuracies
            filename = model_ident_pattern.format(pra,
                                                  tau,
                                                  cfg_model.num_relu_layers,
                                                  cfg_model.neurons_per_layer)
            to_save = {
                "model": model,
                "pra": pra,
                "tau": tau,
            }
            state_dict_fullfilename = os.path.join(out_filedir, filename + ".pt")
            logger.info("Writing {}".format(state_dict_fullfilename))
            torch.save(to_save, state_dict_fullfilename)

    terminal_accuracies = torch.empty((num_taus, num_pras))
    max_accuracies = torch.empty((num_taus, num_pras))

    for t_idx, tau in enumerate(taus):
        for p_idx, pra in enumerate(pras):
            a = accuracies_dict[(tau, pra)]
            terminal_accuracies[t_idx, p_idx] = a[-100:].mean()
            max_accuracies[t_idx, p_idx] = a.max()

    logger.info("{:.4f}".format(max_accuracies.min()))
    logger.info("{:.4f}".format(max_accuracies.mean()))
    logger.info("{:.4f}".format(max_accuracies.median()))

    logger.info("{:.4f}".format(terminal_accuracies.min()))
    logger.info("{:.4f}".format(terminal_accuracies.mean()))
    logger.info("{:.4f}".format(terminal_accuracies.median()))


if __name__ == "__main__":
    main()
