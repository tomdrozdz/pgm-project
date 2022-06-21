import typing as t

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error


def train_forest(
    model: RandomForestClassifier,
    dataset: t.Tuple[
        t.Tuple[torch.Tensor, torch.Tensor], t.Tuple[torch.Tensor, torch.Tensor]
    ],
) -> RandomForestClassifier:
    (X_train, y_train), (X_test, y_test) = dataset

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


def plot_predict_forest(
    model,
    x: pd.DataFrame,
    y: pd.Series,
    start: int = 0,
    stop: t.Optional[int] = None,
):
    if stop is not None:
        stop = len(y)

    x = x.iloc[start:stop]
    y = y.iloc[start:stop]

    y_pred = model.predict(x)

    pd.DataFrame(
        {
            "true": y,
            "pred": y_pred,
        }
    ).plot(lw=0.7, figsize=(10, 5))
