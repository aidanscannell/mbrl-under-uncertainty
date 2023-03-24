#!/usr/bin/env python3
import hydra
from typing import Optional

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
from pytorch_lightning import Trainer
from src.custom_types import (
    Prediction,
    ReplayBuffer,
    ReplayBuffer_to_dynamics_TensorDataset
)
import numpy as np
from src.models import DynamicModel
from torch.utils.data import DataLoader



if __name__ == "__main__":
    import hydra
    # import numpy as np
    import torch

    input_dim = 5
    output_dim = 4

    @hydra.main(version_base="1.3", config_path="../configs", config_name="main")
    def train_gp_dynamic_model(cfg):
        train_x = torch.linspace(-1, 1, 100*input_dim).reshape(-1, input_dim)
        train_y = torch.linspace(1, 3, 100*output_dim).reshape(-1, output_dim)
        print("X.shape {}".format(train_x.shape))
        print("Y.shape {}".format(train_y.shape))
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        model = hydra.utils.instantiate(cfg.model)
        model.train(train_loader)

        Xnew = torch.linspace(2, 3, 100*input_dim).reshape(-1, input_dim)
        Ynew = torch.linspace(3, 4, 100*output_dim).reshape(-1, output_dim)

        Xtest = torch.linspace(3, 4, 7*input_dim).reshape(7, input_dim)

        # Predict WITHOUT fast update
        prediction = model.forward(x=Xtest)
        print("Predict WITHOUT fast update: {}".format(prediction.latent_dist))

        # # Predict WITH fast update
        prediction_with_fast_update = model.forward(x=Xtest, data_new=(Xnew, Ynew))
        print("Predict WITH fast update: {}".format(prediction_with_fast_update.latent_dist))

    #np.assert()
    train_gp_dynamic_model()
