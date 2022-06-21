import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.crf import MLPCRFClassifier, mlpcrf_loss, train_crf
from src.data import batch_dataset, load_data, time_series_split
from src.utils import ensure_reproducibility, print_to_file


def prepare_dataset():
    df = load_data()
    dataset = time_series_split(
        df,
        without_cols=[
            "Timestamp",
            "Light1",
            "Light2",
            "Light3",
            "Light4",
        ],
        scale="normalize",
    )
    return dataset


def run_training(time_series_splits, name, h_size, lr, seq_len):
    losses, f1_macros, mses, f1_micros = [], [], [], []

    for i, split in enumerate(time_series_splits):
        dataset = batch_dataset(split, sequence_length=seq_len)
        i += 1

        print(f"-------------------- SPLIT {i} --------------------")
        mlpcrf_model = MLPCRFClassifier(h_size, 4, 12)
        loss, f1_macro, mse, f1_micro = train_crf(
            model=mlpcrf_model,
            loss_fn=mlpcrf_loss,
            dataset=dataset,
            n_epochs=1000,
            lr=lr,
            batch_size=16,
            use_mask=False,
            patience=50,
            minimum=150,
            name=name,
            split=i,
            save_figs=True,
        )
        print(f"---------------------------------------------------\n\n")

        losses.append(loss)
        f1_macros.append(f1_macro)
        mses.append(mse)
        f1_micros.append(f1_micro)

    print(f"Mean loss: {np.array(losses).mean()}")
    print(f"Mean F1 (macro): {np.array(f1_macros).mean()}")
    print(f"Mean MSE: {np.array(mses).mean()}")
    print(f"Mean F1 (micro): {np.array(f1_micros).mean()}")


def main():
    splits = prepare_dataset()
    ensure_reproducibility()

    hidden_sizes = [[64, 32, 16], [128, 64], [64, 32], [128], [64], [32]]
    learning_rates = [0.001, 0.0007, 0.0003, 0.0001]
    sequence_lengths = [10, 15, 20, 25]

    for h_size, lr, seq_len in product(hidden_sizes, learning_rates, sequence_lengths):
        size_name = "_".join(str(s) for s in h_size)
        lr_name = str(lr).removeprefix("0.")
        seq_name = str(seq_len)

        name = f"{size_name}_{lr_name}_{seq_name}"

        with print_to_file(name):
            start = time.time()
            run_training(splits, name, h_size, lr, seq_len)
            end = time.time()

            print(f"\nTime: {end - start}")

        plt.close("all")


if __name__ == "__main__":
    main()
