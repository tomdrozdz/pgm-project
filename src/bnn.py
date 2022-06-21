import torch
import typing as t
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np


class TwoGaussianMixturePrior:
    def __init__(
            self,
            sigma_1: float = 1,
            sigma_2: float = 1e-6,
            mixing: float = 0.5,
    ):
        self.mixing = mixing

        self.w_prior_1 = torch.distributions.Normal(0, sigma_1)
        self.w_prior_2 = torch.distributions.Normal(0, sigma_2)

        self.b_prior_1 = torch.distributions.Normal(0, sigma_1)
        self.b_prior_2 = torch.distributions.Normal(0, sigma_2)

    def log_prob(self, weights: torch.Tensor, biases: torch.Tensor):
        w_log_prior_1 = self.w_prior_1.log_prob(weights).exp()
        w_log_prior_2 = self.w_prior_2.log_prob(weights).exp()

        w_prior = self.mixing * w_log_prior_1 + (1 - self.mixing) * w_log_prior_2

        b_log_prior_1 = self.b_prior_1.log_prob(biases).exp()
        b_log_prior_2 = self.b_prior_2.log_prob(biases).exp()

        b_prior = self.mixing * b_log_prior_1 + (1 - self.mixing) * b_log_prior_2

        return w_prior.log().mean() + b_prior.log().mean()


class BayesianLinear(nn.Module):
    """Main reference: https://arxiv.org/pdf/1505.05424.pdf"""

    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            prior: TwoGaussianMixturePrior,
    ):
        """Implement initialization of weights and biases values"""
        super().__init__()

        self.prior = prior

        self.last_weights_ = None
        self.last_biases_ = None

        self.weight_mu = nn.Parameter(torch.Tensor(num_input_features, num_output_features))
        self.weight_rho = nn.Parameter(torch.Tensor(num_input_features, num_output_features))

        self.bias_mu = nn.Parameter(torch.Tensor(num_output_features))
        self.bias_rho = nn.Parameter(torch.Tensor(num_output_features))

        self.weight_mu.data.uniform_(-1, 1)
        self.weight_rho.data.uniform_(0, 1)

        self.bias_mu.data.uniform_(-1, 1)
        self.bias_rho.data.uniform_(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implement forward inference using reparametrization trick"""
        weights = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * \
                  torch.zeros_like(self.weight_mu).data.normal_()

        biases = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * \
                 torch.zeros_like(self.bias_mu).data.normal_()

        self.last_weights_ = weights
        self.last_biases_ = biases

        return x @ weights + biases

    def prior_log_prob(self) -> torch.Tensor:
        """Calculates the prior log prob of sampled weights and biases."""
        return self.prior.log_prob(weights=self.last_weights_, biases=self.last_biases_)

    def variational_log_prob(self) -> torch.Tensor:
        """Implement the variational log prob."""
        weights_log_prob = torch.distributions.Normal(self.weight_mu, torch.log1p(torch.exp(self.weight_rho))).log_prob(
            self.last_weights_)
        biases_log_prob = torch.distributions.Normal(self.bias_mu, torch.log1p(torch.exp(self.bias_rho))).log_prob(
            self.last_biases_)
        return weights_log_prob.mean() + biases_log_prob.mean()


class BayesianMLP(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            num_hidden_features: int,
            num_output_classes: int,
            prior: TwoGaussianMixturePrior = None
    ):
        super().__init__()

        self.layer_1 = BayesianLinear(
            num_input_features, num_hidden_features,
            prior=prior or TwoGaussianMixturePrior(),
        )
        self.layer_2 = BayesianLinear(
            num_hidden_features, num_output_classes,
            prior=prior or TwoGaussianMixturePrior(),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sigmoid(self.layer_1(x))
        x = self.softmax(self.layer_2(x))
        return x

    def prior_log_prob(self) -> torch.Tensor:
        log_prob = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                log_prob += module.prior_log_prob()
        return log_prob

    def variational_log_prob(self) -> torch.Tensor:
        log_prob = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                log_prob += module.variational_log_prob()
        return log_prob


class BayesianMLP2(BayesianMLP):
    def __init__(
            self,
            num_input_features: int,
            num_hidden_features: int,
            num_output_classes: int,
            prior: TwoGaussianMixturePrior = None
    ):
        super().__init__(num_input_features, num_hidden_features, num_output_classes, prior)
        self.mid_layer = BayesianLinear(num_hidden_features, num_hidden_features, prior or TwoGaussianMixturePrior())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sigmoid(self.layer_1(x))
        x = self.sigmoid(self.mid_layer(x))
        x = self.softmax(self.layer_2(x))
        return x


class ELBO(nn.Module):
    def __init__(self, N: int):
        super().__init__()

        self.N = N
        self.nll = nn.NLLLoss(reduction="none")

    def forward(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            *,
            return_predictions: bool = False,
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate loss according to the equation (2) of https://arxiv.org/abs/1505.05424"""
        predictions = []
        log_posteriors = []
        log_priors = []
        nlls = []

        for _ in range(self.N):
            preds = model(inputs)

            predictions.append(preds)
            log_posteriors.append(model.variational_log_prob())
            log_priors.append(model.prior_log_prob())
            nlls.append(self.nll(torch.log(preds), targets.argmax(dim=-1)).sum())

        loss = sum(nlls) + sum(log_posteriors) - sum(log_priors)
        loss = loss / self.N

        if return_predictions:
            return loss, torch.stack(predictions, dim=-1)
        return loss


def fit_elbo(
        model: nn.Module,
        dataset: t.Tuple[
            t.Tuple[torch.Tensor, torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]
        ],
        loss_function: nn.Module,
        batch_size: int,
        epochs: int,
        optimizer: torch.optim.Optimizer,
):
    train_metrics = {"loss": [], "acc": [], "step": []}
    test_metrics = {"loss": [], "acc": [], "step": []}

    global_step = 0

    (X_train, y_train), (X_test, y_test) = dataset

    X_train = torch.tensor(X_train.to_numpy()).to(torch.float32)
    X_test = torch.tensor(X_test.to_numpy()).to(torch.float32)

    y_train = one_hot(torch.tensor(y_train.to_numpy()), 4).to(torch.float32)
    y_test = one_hot(torch.tensor(y_test.to_numpy()), 4).to(torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")

        model.train()
        pbar = tqdm(train_dataloader)
        for inputs, targets in pbar:
            optimizer.zero_grad()

            loss, y_predictions = loss_function(
                model, inputs, targets, return_predictions=True
            )
            loss.backward()
            optimizer.step()

            y_predictions = y_predictions.mean(dim=-1)
            accuracy = f1_score(targets.argmax(dim=1), y_predictions.argmax(dim=1), average='macro')

            train_metrics["loss"].append(loss.item())
            train_metrics["acc"].append(accuracy.item())
            train_metrics["step"].append(global_step)
            global_step += 1
            pbar.update(1)
        pbar.close()

        model.eval()

        preds = []
        trues = []
        total_batches = 0
        total_loss = 0.0
        for inputs, targets in test_dataloader:
            loss, y_predictions = loss_function(
                model, inputs, targets, return_predictions=True
            )
            y_predictions = y_predictions.mean(dim=-1)
            total_batches += 1

            total_loss += loss.item()

            trues.append(targets)
            preds.append(y_predictions)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        val_acc = f1_score(trues.argmax(dim=1), preds.argmax(dim=1), average='macro')

        test_metrics["loss"].append(total_loss / total_batches)
        test_metrics["acc"].append(val_acc)
        test_metrics["step"].append(global_step)

    return train_metrics, test_metrics


def plot_predict_bnn(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        start: int = 0,
        stop=None,
        n_samples=100
):
    if stop is not None:
        stop = len(y)

    with torch.no_grad():
        y_pred = model(x).argmax(dim=1)[start:stop]
        y_test = y[start:stop]

        out_arr = torch.zeros((y_pred.shape[0], 4))
        for i in range(n_samples):
            out = model(x)
            out_oneh = one_hot(torch.argmax(out, dim=1), num_classes=4)
            out_arr[:, :] += out_oneh

        outs = out_arr.numpy() / n_samples

    pd.DataFrame(
        {
            "true": y_test,
            "pred": y_pred,
            'proba': np.max(outs, axis=1),
        }
    ).plot(lw=0.7, figsize=(10, 5))


def show_learning_curve(
        train_metrics: t.Dict[str, t.List[float]], test_metrics: t.Dict[str, t.List[float]]
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(train_metrics["step"], train_metrics["loss"], label="train")
    ax.plot(test_metrics["step"], test_metrics["loss"], label="test")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve")
    plt.legend()


def show_accuracy_curve(
        train_metrics: t.Dict[str, t.List[float]], test_metrics: t.Dict[str, t.List[float]]
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(train_metrics["step"], train_metrics["acc"], label="train")
    ax.plot(test_metrics["step"], test_metrics["acc"], label="test")
    ax.set_xlabel("Training step")
    ax.set_ylabel("F1")
    ax.set_title("F1 curve")
    plt.legend()
