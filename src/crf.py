import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import load_model_state, save_fig, save_model


class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).

    based on: https://github.com/mtreviso/linear-chain-crf/blob/master/crf_vectorized.py

    Args:
        nb_labels (int): number of labels in your dataset.
    """

    def __init__(self, nb_labels):
        super().__init__()

        self.nb_labels = nb_labels
        self.PAD_ID = -1

        self.transitions = nn.Parameter(
            torch.empty(self.nb_labels + 1, self.nb_labels + 1)
        )
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[self.PAD_ID, :] = -10000.0
        self.transitions.data[:, self.PAD_ID] = -10000.0

    def forward(self, x, mask=None):
        """Select best path for given data."""
        return self.decode(x, mask=mask)

    def log_likelihood(self, emissions, labels, mask=None):
        """Compute the probability of a sequence of labels given a sequence of
        emissions scores.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels).
            labels (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len).
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len).
        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, labels, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return scores - partition

    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels).
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len).
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)  # .cuda()

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, labels, mask):
        """Compute the scores for a given batch of emissions with their labels.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            labels (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = labels.shape
        scores = torch.zeros(batch_size)  # .cuda()

        current = labels[:, 0]

        # add the [unary] emission scores for the first labels for each batch
        # for all batches, the first label, see the correspondent emissions
        # for the first labels (which is a list of ids):
        # emissions[:, 0, [label_1, label_2, ..., label_nblabels]]
        scores += emissions[:, 0].gather(1, current.unsqueeze(1)).squeeze()

        # now lets do this for each remaining labels
        for i in range(1, seq_length):
            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i].int()

            previous = current
            current = labels[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous, current]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        alphas = emissions[:, 0]

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1).int()
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # return a *log* of sums of exps
        return torch.logsumexp(alphas, dim=1)

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            torch.Tensor: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape

        alphas = emissions[:, 0]

        backpointers = []

        for i in range(1, seq_length):
            # (bs, nb_labels) -> (bs, 1, nb_labels)
            e_scores = emissions[:, i].unsqueeze(1)

            # (nb_labels, nb_labels) -> (bs, nb_labels, nb_labels)
            t_scores = self.transitions.unsqueeze(0)

            # (bs, nb_labels)  -> (bs, nb_labels, 1)
            a_scores = alphas.unsqueeze(2)

            # combine current scores with previous alphas
            scores = e_scores + t_scores + a_scores

            # so far is exactly like the forward algorithm,
            # but now, instead of calculating the logsumexp,
            # we will find the highest score and the label associated with it
            max_scores, max_score_labels = torch.max(scores, dim=1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas

            # add the max_score_labels for our list of backpointers
            # max_scores has shape (batch_size, nb_labels) so we transpose it to
            # be compatible with our previous loopy version of viterbi
            backpointers.append(max_score_labels.t())

        # get the final most probable score and the final most probable label
        max_final_scores, max_final_labels = torch.max(alphas, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max label for the last timestep
            sample_final_label = max_final_labels[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_label
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(
                i, sample_final_label, sample_backpointers
            )

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, torch.tensor(best_sequences)

    def _find_best_path(self, sample_id, best_label, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.
        Args:
            sample_id (int): sample index in the range [0, batch_size)
            best_label (int): label which maximizes the final score
            backpointers (list of lists of tensors): list of pointers with
            shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
            represents the length of the ith sample in the batch
        Returns:
            list of ints: a list of label indexes representing the bast path
        """
        # add the final best_label to our best path
        best_path = [best_label]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):
            # recover the best_label at this timestep
            best_label = backpointers_t[best_label][sample_id].item()
            best_path.append(best_label)

        return best_path[::-1]


class MLPCRFClassifier(nn.Module):
    def __init__(self, hidden_sizes: t.List[int], nb_labels: int, x_size: int):
        super().__init__()
        self.nb_labels = nb_labels
        self.x_size = x_size

        self.crf = CRF(self.nb_labels)

        hidden_sizes = [self.x_size, *(hidden_sizes or [])]
        layers: t.List[nn.Module] = []

        for in_size, out_size in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(out_size, self.nb_labels + 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(*x.shape[:2], -1)
        batch_size, seq_length, h_size = x.shape
        x = x.view(-1, h_size)

        x = self.mlp(x)

        return x.view(batch_size, seq_length, self.nb_labels + 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        emissions = self.forward(x)
        _, prediction = self.crf.decode(emissions)
        return prediction


def mlpcrf_loss(
    emissions: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    model: MLPCRFClassifier,
) -> torch.Tensor:
    """Loss for evaluating MLP-CRF Classifier performance."""
    if mask is None:
        mask = torch.ones(y_true.shape[:2], dtype=torch.bool)  # .cuda()

    # use emissions and ground truth to calculate the loss function
    loss = -model.crf.log_likelihood(emissions, y_true, mask).mean()

    return loss


def get_mask(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get mask for padding by examining contents of the X tensor."""
    pad_id = torch.max(y).int().item()
    flattened_X = X.view(*X.shape[:2], -1)
    mask = (torch.eq(flattened_X, pad_id).sum(-1) != flattened_X.shape[-1]).bool()
    return mask


def crf_evaluate(model, loss_fn, dl, use_mask):
    mask = None
    pred_list = []
    true_list = []
    losses = []

    with torch.no_grad():
        for X, y in dl:
            if use_mask:
                mask = get_mask(X, y)

            y_pred = model(X)
            loss = loss_fn(y_pred, y, mask, model)
            losses.append(loss.item())

            pred_list.append(model.predict(X).view(-1))
            true_list.append(y.view(-1))

        pred = torch.cat(pred_list).cpu()
        true = torch.cat(true_list).cpu()

    return np.mean(losses), true, pred


def plot_predict_crf(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    start: int = 0,
    stop: t.Optional[int] = None,
):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))

    if stop is None:
        stop = len(y)

    with torch.no_grad():
        y_pred = model.predict(x).flatten()[start:stop]
        y_test = y.flatten()[start:stop]

    pd.DataFrame(
        {
            "true": y_test.cpu(),
            "pred": y_pred.cpu(),
        }
    ).plot(lw=0.7, ax=ax)

    plt.tight_layout()
    return fig


def plot_transitions(transitions: t.List[npt.NDArray]):
    """Display how transitions weights changed throughout training.

    .. note: we limit ourselves to only 7 first labels for the sake of performance
        and do not consider padding axis (it is not trained)
    """
    size = min(transitions[0].shape[0] - 1, 7)
    fig, axes = plt.subplots(size, size, sharex=True, sharey=True, figsize=(7, 7))

    x = list(range(len(transitions)))
    t = np.array(transitions).transpose(1, 2, 0)[:size, :size]

    for ax_row, val_row in zip(axes, t):
        for ax, val in zip(ax_row, val_row):
            ax.plot(x, val)

    for i, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(i)

    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(i)

    plt.tight_layout()
    return fig


def train_crf(
    model: MLPCRFClassifier,
    loss_fn: t.Callable[
        [torch.Tensor, torch.Tensor, t.Optional[torch.Tensor], MLPCRFClassifier],
        torch.Tensor,
    ],
    dataset: t.Tuple[
        t.Tuple[torch.Tensor, torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]
    ],
    n_epochs: int = 1000,
    lr: float = 0.001,
    batch_size: int = 32,
    use_mask: bool = False,
    patience: int = 50,
    minimum: int = 150,
    name: str = "model",
    split: int = 0,
    save_figs: bool = False,
) -> t.Tuple[float, float, float, float]:
    (X_train, y_train), (X_test, y_test) = dataset
    mask = None
    best_loss = float("+inf")
    loss_increases = 0

    # X_train, y_train = X_train.cuda(), y_train.cuda()
    # X_test, y_test = X_test.cuda(), y_test.cuda()

    train_dataset = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log: t.Dict[str, t.List] = defaultdict(list)

    for epoch in range(1, n_epochs + 1):
        train_pred_list = []
        train_true_list = []
        train_losses = []

        model.zero_grad()
        # train
        for X, y in train_dl:
            if use_mask:
                mask = get_mask(X, y)

            y_pred = model(X)
            loss = loss_fn(y_pred, y, mask, model)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().item())
            with torch.no_grad():
                train_true_list.append(y.view(-1))
                train_pred_list.append(model.predict(X).view(-1))

        # eval
        test_loss, test_true, test_pred = crf_evaluate(
            model, loss_fn=loss_fn, dl=test_dl, use_mask=use_mask
        )

        train_pred = torch.cat(train_pred_list).cpu()
        train_true = torch.cat(train_true_list).cpu()

        # log
        log["epoch"].append(epoch)
        log["train_loss"].append(np.mean(train_losses))
        log["test_loss"].append(test_loss)
        log["train_f1"].append(f1_score(train_true, train_pred, average="macro"))
        log["test_f1"].append(f1_score(test_true, test_pred, average="macro"))
        log["train_mse"].append(mean_squared_error(train_true, train_pred))
        log["test_mse"].append(mean_squared_error(test_true, test_pred))

        trans: npt.NDArray = model.crf.transitions.detach().cpu().numpy()
        log["transitions"].append(trans.copy())

        if test_loss > best_loss:
            loss_increases += 1
        else:
            loss_increases = 0
            best_loss = test_loss

        if loss_increases >= patience and epoch >= minimum or epoch == n_epochs:
            epoch_to_load = epoch - loss_increases

            if epoch_to_load != n_epochs:
                load_model_state(model, epoch_to_load, name)
                print(f"Test loss increased {loss_increases} times, early stopping...")
                print(f"Loading model from epoch {epoch_to_load}\n")
                break
        else:
            save_model(model, epoch, name)

    test_loss, test_true, test_pred = crf_evaluate(
        model, loss_fn=loss_fn, dl=test_dl, use_mask=use_mask
    )

    f1_macro = f1_score(test_true, test_pred, average="macro")
    mse = mean_squared_error(test_true, test_pred)
    f1_micro = f1_score(test_true, test_pred, average="micro")

    print(f"Test loss: {test_loss}")
    print(f"Test F1 (macro): {f1_macro}")
    print(f"Test MSE: {mse}")
    print(f"Test F1 (micro): {f1_micro}\n")

    print(classification_report(test_true, test_pred, zero_division=0))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    df = pd.DataFrame(log)
    df.plot(x="epoch", y=["train_loss", "test_loss"], ax=axes[0])
    df.plot(x="epoch", y=["train_mse", "test_mse"], ax=axes[1])
    df.plot(x="epoch", y=["train_f1", "test_f1"], ax=axes[2])
    plt.tight_layout()

    if save_figs:
        save_fig(fig, name, f"training_{split}")
        save_fig(plot_predict_crf(model, X_test, y_test), name, f"predictions_{split}")
        save_fig(plot_transitions(log["transitions"]), name, f"transitions_{split}")
    else:
        plt.show()
        plot_predict_crf(model, X_test, y_test)
        plt.show()
        plot_transitions(log["transitions"])
        plt.show()

    return test_loss, f1_macro, mse, f1_micro
