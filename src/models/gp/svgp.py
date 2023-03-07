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
from copy import deepcopy
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import (
    DiagLazyTensor,
    CholLazyTensor,
    TriangularLazyTensor,
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
            batch_size: int = 16,
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
            out_size=covar_module.batch_shape[0],
            # TODO is this the bset way to set out_size? Not like this!
        )
        if trainer is None:
            self.trainer = Trainer()
        else:
            self.trainer = trainer
        self.batch_size = batch_size
        self.delta_state = delta_state
        self.num_workers = num_workers

    def forward(self, x, data_new: Optional = None) -> Prediction:

        if data_new == None:

            # self.gp_module.eval()
            f = self.gp_module.forward(x, data_new=data_new)
            print("latent {}".format(f.variance))

            output = self.gp_module.likelihood(f)
            print("output {}".format(output))
            f_dist = td.Normal(loc=f.mean, scale=torch.sqrt(f.variance))
            print("f_dist {}".format(f_dist))
            y_dist = td.Normal(loc=f.mean, scale=torch.sqrt(output.variance))
            print("y_dist {}".format(y_dist))
            noise_var = output.variance - f.variance
        # pred = Prediction(latent=f_dist, output=y_dist, noise_var=noise_var)

        else:
            X, Y = data_new

            # # make copy of self
            model = self.make_copy()
            inducing_points = self.gp_module.inducing_points

            var_cov_root = TriangularLazyTensor(
                self.variational_strategy._variational_distribution.chol_variational_covar
            )
            var_cov = CholLazyTensor(var_cov_root)
            var_mean = (
                self.variational_strategy.variational_distribution.mean
            )  # .unsqueeze(-1)
            if var_mean.shape[-1] != 1:  # TODO: won't work for M=1 ...
                var_mean = var_mean.unsqueeze(-1)

            # var_dist = self.variational_strategy.variational_distribution
            # var_mean = var_dist.mean
            # var_cov = var_dist.lazy_covariance_matrix

            # GPyTorch's way of computing Kuf:
            # full_inputs = torch.cat([inducing_points, X], dim=-2)
            full_inputs = torch.cat([torch.tile(inducing_points, X.shape[:-2] + (1, 1)), X], dim=-2)
            full_covar = self.covar_module(full_inputs)

            # Covariance terms
            num_induc = inducing_points.size(-2)
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()

            K_uf = induc_data_covar

            # Kuu = self.covar_module(inducing_points)
            Kuu = induc_induc_covar
            # Kuu_root = Kuu.cholesky()

            lambda_1, lambda_2 = mean_cov_to_natural_param(var_mean, var_cov, Kuu)

            lambda_1_t = torch.zeros_like(lambda_1)
        # lambda_2_t = torch.zeros_like(lambda_2)
        #
        # # online_update
        # for _ in range(self.num_online_updates):
        #     # grad_varexp_natural_params
        #     with torch.no_grad():
        #         Xt = torch.tile(X, Y.shape[:-2] + (1, 1, 1))
        #         #                 if Y.shape[-1] == 1:
        #         #                     Xt.unsqueeze_(-1)
        #         pred = fantasy_model(Xt)
        #         mean = pred.mean
        #         var = pred.variance
        #     mean.requires_grad_()
        #     var.requires_grad_()
        #
        #     # variational expectations
        #     f_dist = MultivariateNormal(mean, DiagLazyTensor(var))
        #     ve_terms = fantasy_model.likelihood.expected_log_prob(Y, f_dist)
        #     ve = ve_terms.sum()  # TODO: CHECK: divide by num_data ? but only one point at a time so probably fine
        #
        #     ve.backward(inputs=[mean, var])
        #     d_exp_dm = mean.grad  # [batch, N]
        #     d_exp_dv = var.grad  # [batch, N]
        #
        #     eps = 1e-8
        #     d_exp_dv.clamp_(max=-eps)
        #
        #     grad_nat_1 = (d_exp_dm - 2.0 * (d_exp_dv * mean))
        #     grad_nat_2 = d_exp_dv
        #
        #     grad_mu_1 = K_uf.matmul(grad_nat_1[..., None])
        #
        #     grad_mu_2 = K_uf.matmul(DiagLazyTensor(grad_nat_2).matmul(K_uf.swapdims(-1, -2)))
        #
        #     lr = self.lr
        #     scale = 1.0
        #
        #     lambda_1_t_new = (1.0 - lr) * lambda_1_t + lr * scale * grad_mu_1
        #     lambda_2_t_new = (1.0 - lr) * lambda_2_t + lr * scale * (-2) * grad_mu_2
        #
        #     lambda_1_new = lambda_1 - lambda_1_t + lambda_1_t_new
        #     lambda_2_new = lambda_2 - lambda_2_t + lambda_2_t_new
        #
        #     new_mean, new_cov = conditional_from_precision_sites_white_full(
        #         Kuu, lambda_1_new, lambda_2_new,
        #         jitter=getattr(self, "tsvgp_jitter", 0.0)
        #     )
        #     new_mean = new_mean.squeeze(-1)
        #     new_cov_root = new_cov.cholesky()
        #
        #     fantasy_var_dist = fantasy_model.variational_strategy._variational_distribution
        #     with torch.no_grad():
        #         fantasy_var_dist.variational_mean.set_(new_mean)
        #         fantasy_var_dist.chol_variational_covar.set_(new_cov_root)
        #
        #     lambda_1 = lambda_1_new
        #     lambda_2 = lambda_2_new
        #     lambda_1_t = lambda_1_t_new
        #     lambda_2_t = lambda_2_t_new
        #
        pred = Prediction(latent_dist=f_dist, output_dist=y_dist, noise_var=noise_var)
        return pred

    @property
    def model(self):
        return self.gp_module.gp

    def make_copy(self):
        with torch.no_grad():
            inducing_points = self.model.variational_strategy.base_variational_strategy.inducing_points.detach().clone()

            if hasattr(self, "input_transform"):
                [p.detach_() for p in self.input_transform.buffers()]

            #new_covar_module = deepcopy(self.gp_module.covar_module)

            new_model = self.__class__(
                likelihood=deepcopy(self.gp_module.likelihood),
                mean_module=deepcopy(self.model.mean_module),
                covar_module=deepcopy(self.model.covar_module),
                learning_rate=deepcopy(self.gp_module.learning_rate),
            )
            #             new_model.mean_module = deepcopy(self.mean_module)
            #             new_model.likelihood = deepcopy(self.likelihood)

            var_dist = self.model.variational_strategy.base_variational_strategy.variational_distribution
            mean = var_dist.mean.detach().clone()
            cov = var_dist.covariance_matrix.detach().clone()

            new_var_dist = new_model.model.variational_strategy.base_variational_strategy.variational_distribution
            with torch.no_grad():
                new_var_dist.mean.set_(mean)
                new_var_dist.covariance_matrix.set_(cov)
                new_model.model.variational_strategy.base_variational_strategy.inducing_points.set_(inducing_points)

            new_model.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(1)

        return new_model

    # def train(self, replay_buffer: ReplayBuffer):
    #     dataset = ReplayBuffer_to_dynamics_TensorDataset(
    #         replay_buffer, delta_state=self.delta_state
    #     )
    def train(self, dataset):
        train_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=self.batch_size,
                # batch_size=num_data,
                drop_last=True,
            ),
            num_workers=self.num_workers,
        )

        # self.gp_module.set_train_data(inputs=train_x, targets=train_y, strict=False)
        print("before FIT")
        self.gp_module.gp.train()
        self.gp_module.likelihood.train()
        self.trainer.fit(self.gp_module, train_dataloaders=train_loader)
        print("after FIT")
        self.gp_module.gp.eval()
        self.gp_module.likelihood.eval()


def mean_cov_to_natural_param(mu, Su, K_uu):
    """
    Transforms (m,S) to (λ₁,P) tsvgp_white parameterization
    """
    lamb1 = K_uu.matmul(Su.inv_matmul(mu))
    lamb2 = K_uu.matmul(Su.inv_matmul(K_uu.evaluate())) - K_uu.evaluate()

    return lamb1, lamb2


def conditional_from_precision_sites_white_full(
        Kuu,
        lambda1,
        Lambda2,
        jitter=1e-9,
):
    """
    Given a g₁ and g2, and distribution p and q such that
      p(g₂) = N(g₂; 0, Kuu)
      p(g₁) = N(g₁; 0, Kff)
      p(g₁ | g₂) = N(g₁; Kfu (Kuu⁻¹) g₂, Kff - Kfu (Kuu⁻¹) Kuf)
    And  q(g₂) = N(g₂; μ, Σ) such that
        Σ⁻¹  = Kuu⁻¹  + Kuu⁻¹LLᵀKuu⁻¹
        Σ⁻¹μ = Kuu⁻¹l
    This method computes the mean and (co)variance of
      q(g₁) = ∫ q(g₂) p(g₁ | g₂) dg₂ = N(g₂; μ*, Σ**)
    with
    Σ** = k** - kfu Kuu⁻¹ kuf - kfu Kuu⁻¹ Σ Kuu⁻¹ kuf
        = k** - kfu Kuu⁻¹kuf + kfu (Kuu + LLᵀ)⁻¹ kuf
    μ* = k*u Kuu⁻¹ m
       = k*u Kuu⁻¹ Λ⁻¹ Kuu⁻¹ l
       = k*u (Kuu + LLᵀ)⁻¹ l
    Inputs:
    :param Kuu: tensor M x M
    :param l: tensor M x 1
    :param L: tensor M x M
    """
    # TODO: rewrite this

    R = (Lambda2 + Kuu).add_jitter(jitter)

    mean = Kuu.matmul(R.inv_matmul(lambda1))
    cov = Kuu.matmul(R.inv_matmul(Kuu.evaluate()))  # TODO: symmetrise?
    return mean, cov


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
                        # num_inducing_points=num_inducing,
                        num_inducing_points=inducing_points.size(-2),
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

            def forward(self, x, data_new: Optional = None):
                if data_new is None:
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                else:
                    raise NotImplementedError("# TODO Paul implement fast update here")
                # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                    )
                )

        self.gp = IndependentMultitaskGPModel()
        self.likelihood = likelihood

    def forward(self, x, data_new: Optional = None):
        # def forward(self, x):
        return self.gp.forward(x, data_new=data_new)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.gp.forward(x)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=y.size(0)
        )
        loss = -mll(y_pred, y)
        self.log("train_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.gp.parameters(), lr=self.learning_rate)
