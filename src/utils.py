import random
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
import torch

MODELS_PATH = Path("models")
RESULTS_PATH = Path("results")


def save_model(model, epoch, name):
    MODELS_PATH.mkdir(exist_ok=True)

    path = MODELS_PATH / name
    path.mkdir(exist_ok=True)

    torch.save(
        obj={"model_state_dict": model.state_dict()},
        f=str(path / f"epoch_{epoch}"),
    )


def load_model_state(model, epoch, name):
    checkpoint = torch.load(str(MODELS_PATH / name / f"epoch_{epoch}"))
    model.load_state_dict(checkpoint["model_state_dict"])


def save_fig(fig, model_name: str, fig_name: str):
    RESULTS_PATH.mkdir(exist_ok=True)

    path = RESULTS_PATH / model_name
    path.mkdir(exist_ok=True)

    fig.savefig(str(path / fig_name), dpi=300, bbox_inches="tight")


@contextmanager
def print_to_file(name: str):
    RESULTS_PATH.mkdir(exist_ok=True)

    path = RESULTS_PATH / name
    path.mkdir(exist_ok=True)

    path = path / f"output.txt"

    with path.open("w") as f:
        with redirect_stdout(f) as ctx:
            yield ctx


def ensure_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
