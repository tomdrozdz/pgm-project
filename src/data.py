import datetime as dt
import typing as t

import numpy as np
import pandas as pd
import torch


def load_data(path: str = "Occupancy_Estimation.csv") -> pd.DataFrame:
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


def drop_data(
    data: pd.DataFrame, to_drop: t.List[t.Tuple[dt.datetime, dt.datetime]] = None
):
    if to_drop is None:
        # Wyrzucane są noce + w święta między 24 a 27 nikogo nie było
        to_drop = [
            (dt.datetime(2017, 12, 22, 21), dt.datetime(2017, 12, 23, 9)),
            (dt.datetime(2017, 12, 23, 21), dt.datetime(2017, 12, 27)),
            (dt.datetime(2018, 1, 10, 21), data["Timestamp"].max()),
        ]

    for start, end in to_drop:
        indices = (data["Timestamp"] >= start) & (data["Timestamp"] < end)
        data = data[~indices]

    data = data.reset_index(drop=True)

    return data


def _scale(x_train: pd.DataFrame, x_test: pd.DataFrame, scale: t.Optional[str]) -> None:
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


def split_data(
    data: pd.DataFrame,
    without_cols: t.Optional[t.List[str]] = None,
    scale: t.Optional[str] = None,
) -> t.Tuple[t.Tuple[pd.DataFrame, pd.Series], t.Tuple[pd.DataFrame, pd.Series]]:
    if without_cols is None:
        without_cols = ["Timestamp", "Light1", "Light2", "Light3", "Light4"]

    df = data[data["Timestamp"].dt.date != data["Timestamp"].dt.date.max()]
    x, y = df.drop(columns=without_cols + ["Occupancy"]), df["Occupancy"]

    train_index = df["Timestamp"].dt.date != df["Timestamp"].dt.date.max()
    test_index = df["Timestamp"].dt.date == df["Timestamp"].dt.date.max()

    x_train = x[train_index]
    x_test = x[test_index]

    _scale(x_train, x_test, scale)

    return (x_train, y[train_index]), (x_test, y[test_index])


def time_series_split(
    data: pd.DataFrame,
    without_cols: t.Optional[t.List[str]] = None,
    scale: t.Optional[str] = None,
    split_points: t.Optional[t.List[int]] = None,
) -> t.List[
    t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]]
]:
    if without_cols is None:
        without_cols = ["Timestamp", "Light1", "Light2", "Light3", "Light4"]

    if split_points is None:
        if len(data) > 5000:
            split_points = [475, 2775, 3200, 8000]
        else:
            split_points = [475, 1325, 1725, 2270]

    df = data[data["Timestamp"].dt.date != data["Timestamp"].dt.date.max()]
    x, y = df.drop(columns=without_cols + ["Occupancy"]), df["Occupancy"]

    series = []
    split_points = split_points + [len(data)]

    for middle, end in zip(split_points, split_points[1:]):
        (x_train, y_train) = x.iloc[0:middle], y.iloc[0:middle]
        (x_test, y_test) = x.iloc[middle:end], y.iloc[middle:end]

        _scale(x_train, x_test, scale)

        series.append(((x_train, y_train), (x_test, y_test)))

    return series


def _batch(x: pd.DataFrame, y: pd.Series, seq_len: int, overlap_series: bool):
    xs = []
    ys = []

    if overlap_series:
        xs = [x.iloc[i : i + seq_len].to_numpy() for i in range(len(x) - seq_len)]
        ys = [y.iloc[i : i + seq_len].to_numpy() for i in range(len(x) - seq_len)]
    else:
        bot = 0
        top = 0 + seq_len
        while top <= len(x):
            xs.append(x.iloc[bot:top].to_numpy())
            ys.append(y.iloc[bot:top].to_numpy())
            bot += seq_len
            top += seq_len

    return torch.from_numpy(np.array(xs)).float(), torch.from_numpy(np.array(ys))


def batch_dataset(
    dataset: t.Tuple[
        t.Tuple[pd.DataFrame, pd.Series], t.Tuple[pd.DataFrame, pd.Series]
    ],
    sequence_length: int = 25,
    overlap_series: bool = False,
) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]]:
    (x_train, y_train), (x_test, y_test) = dataset

    x_tr, y_tr = _batch(
        x_train, y_train, seq_len=sequence_length, overlap_series=overlap_series
    )
    x_te, y_te = _batch(
        x_test, y_test, seq_len=sequence_length, overlap_series=overlap_series
    )

    return (x_tr, y_tr), (x_te, y_te)
