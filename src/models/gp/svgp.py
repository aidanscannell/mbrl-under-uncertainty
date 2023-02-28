#!/usr/bin/env python3
from typing import Optional

import gpytorch
import pytorch_lightning as pl
import torch
import torch.distributions as td
from pytorch_lightning import Trainer
from src.custom_types import (
    Action,
    Prediction,
    ReplayBuffer,
    ReplayBuffer_to_dynamics_TensorDataset,
    State,
)
from src.models import DynamicModel
from torch.utils.data import DataLoader, TensorDataset


class SVGPDynamicModel(DynamicModel):
    # TODO implement delta_state properly
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        learning_rate: float = 0.1,
        trainer: Optional[Trainer] = None,
        delta_state: bool = True,
        num_workers: int = 1,
    ):
        super(SVGPDynamicModel, self).__init__()
        self.gp_module = SVGPModule(
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            learning_rate=learning_rate,
        )
        if trainer is None:
            self.trainer = Trainer()
        else:
            self.trainer = trainer
        self.delta_state = delta_state
        self.num_workers = num_workers

    def forward(self, x) -> Prediction:
        self.gp_module.eval()
        latent = self.gp_module(x)
        print("latent")
        print(latent)
        output = self.gp_module.likelihood(latent)
        print("output")
        print(output)
        f_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(latent.variance))
        print("f_dist")
        print(f_dist)
        y_dist = td.Normal(loc=latent.mean, scale=torch.sqrt(output.variance))
        print("y_dist")
        print(y_dist)
        noise_var = output.variance - latent.variance
        print("noise_var")
        print(noise_var)
        # pred = Prediction(latent=f_dist, output=y_dist, noise_var=noise_var)
        pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
        print("pred")
        print(pred)
        return pred

    def train(self, replay_buffer: ReplayBuffer):
        dataset = ReplayBuffer_to_dynamics_TensorDataset(
            replay_buffer, delta_state=self.delta_state
        )
        train_x = dataset.tensors[0]
        train_y = dataset.tensors[1]
        num_data = train_x.shape[0]
        train_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=num_data,
                drop_last=True,
            ),
            num_workers=self.num_workers,
        )

        self.gp_module.set_train_data(inputs=train_x, targets=train_y, strict=False)
        print("before FIT")
        self.gp_module.gp.train()
        self.gp_module.likelihood.train()
        self.trainer.fit(self.gp_module, train_dataloaders=train_loader)
        print("after FIT")
        self.gp_module.gp.eval()
        self.gp_module.likelihood.eval()


class SVGPModule(pl.LightningModule):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        covar_module: gpytorch.kernels.Kernel = None,
        num_inducing: int = 16,
        out_size: int = 1,
        learning_rate: float = 1e-3,
    ):
        super(SVGPModule, self).__init__()
        self.learning_rate = learning_rate

        # Learn seperate hyperparameters for each output dim
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=out_size
            )
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([out_size])
            )
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([out_size])),
                batch_shape=torch.Size([out_size]),
            )

        class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
            def __init__(self):
                inducing_points = torch.rand(out_size, num_inducing, 1)

                # Learn a variational distribution for each output dim
                variational_distribution = (
                    gpytorch.variational.CholeskyVariationalDistribution(
                        num_inducing_points=num_inducing,
                        batch_shape=torch.Size([out_size]),
                    )
                )
                variational_strategy = (
                    gpytorch.variational.IndependentMultitaskVariationalStrategy(
                        gpytorch.variational.VariationalStrategy(
                            self,
                            inducing_points,
                            variational_distribution,
                            learn_inducing_locations=True,
                        ),
                        num_tasks=out_size,
                    )
                )

                super().__init__(variational_strategy)
                self.mean_module = mean_module
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.gp = IndependentMultitaskGPModel()
        self.likelihood = likelihood

    def forward(self, x):
        return self.gp.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.gp.forward(x)
        print("y_pred.shape {}".format(y_pred.shape))
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=y.size(0)
        )
        loss = -mll(y_pred, y)
        self.log("train_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.gp.parameters(), lr=self.learning_rate)
