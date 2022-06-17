import typing as t

import numpy as np
import pandas as pd
import torch


def load_data(path="Occupancy_Estimation.csv") -> pd.DataFrame:
    data = pd.read_csv(
        path,
        names=[
            "Date",
            "Time",
            "Temp1",
            "Temp2",
            "Temp3",
            "Temp4",
            "Light1",
            "Light2",
            "Light3",
            "Light4",
            "Sound1",
            "Sound2",
            "Sound3",
            "Sound4",
            "CO2",
            "CO2_Slope",
            "Motion1",
            "Motion2",
            "Occupancy",
        ],
        header=0,
    )

    data["Timestamp"] = pd.to_datetime(data["Date"] + " " + data["Time"])
    data.drop(columns=["Date", "Time"], inplace=True)

    return data


def split_data(data, without_cols: t.List[str], scale: t.Optional[str] = None):
    df = data[data["Timestamp"].dt.date != data["Timestamp"].dt.date.max()]
    x, y = df.drop(columns=without_cols + ["Occupancy"]), df["Occupancy"]

    train_index = df["Timestamp"].dt.date != df["Timestamp"].dt.date.max()
    test_index = df["Timestamp"].dt.date == df["Timestamp"].dt.date.max()

    x_train = x[train_index]
    x_test = x[test_index]

    if scale == "normalize":
        mins = x_train.min()
        maxs = x_train.max()
        denom = maxs - mins

        x_train = (x_train - mins) / denom
        x_test = (x_test - mins) / denom
    elif scale == "standardize":
        means = x_train.mean()
        stds = x_train.std()

        x_train = (x_train - means) / stds
        x_test = (x_test - means) / stds
    elif scale is not None:
        raise ValueError(f"'scale' can be equal to 'normalize', 'standardize' or None")

    return (x_train, y[train_index]), (x_test, y[test_index])


def _batch(x, y, seq_len, overlap_series):
    xs = []
    ys = []

    if overlap_series:
        xs = [x.iloc[i:i+seq_len].to_numpy() for i in range(0, len(x)-seq_len)]
        ys = [y.iloc[i:i+seq_len].to_numpy() for i in range(0, len(x)-seq_len)]
    else:
        bot = 0
        top = 0 + seq_len
        while top <= len(x):
            xs.append(x.iloc[bot:top].values)
            ys.append(y.iloc[bot:top].values)
            bot += seq_len
            top += seq_len

    return torch.from_numpy(np.array(xs)).float(), torch.from_numpy(np.array(ys))


def batch_dataset(dataset, sequence_length=25, overlap_series=False):
    (x_train, y_train), (x_test, y_test) = dataset

    x_tr, y_tr = _batch(x_train, y_train, seq_len=sequence_length, overlap_series=overlap_series)
    x_te, y_te = _batch(x_test, y_test, seq_len=sequence_length, overlap_series=overlap_series)

    return (x_tr, y_tr), (x_te, y_te)
