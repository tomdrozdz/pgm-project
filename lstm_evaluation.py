import sys

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.data import time_series_split, load_data, batch_dataset
from src.lstm import visualise_clf_results
from src.lstm import visualise_regr_results
from src.lstm import get_best_param_model
import torch
import numpy as np

OUT_TO_FILE = True
PARAMS_TUNING_RESULTS_FILE = 'hp_tuning_2.pkl'

torch.manual_seed(11)
np.random.seed(11)


def evaluate_mcdlstm(df, regression, n_splits=4):
    f1_vals = []
    mse_vals = []
    clf_reports = []
    f1_vals_train = []
    mse_vals_train = []
    for i in range(0, n_splits):
        print(f'Evaluating for split {i+1} of {n_splits}')
        series = time_series_split(df)
        (x_train, y_train), (x_test, y_test) = series[i]
        scaler = MinMaxScaler()
        (x_train, y_train), (x_test, y_test) = \
            (pd.DataFrame(scaler.fit_transform(x_train)), y_train), \
            (pd.DataFrame(scaler.transform(x_test)), y_test)
        device='cuda:0'
        model, seq_len, f1_train, mse_train = get_best_param_model(PARAMS_TUNING_RESULTS_FILE, regression, i, device)
        f1_vals_train.append(f1_train)
        mse_vals_train.append(mse_train)
        model = model.to(device)
        (x_train, y_train), (x_test, y_test) = batch_dataset(((x_train, y_train), (x_test, y_test)), seq_len, True)

        plt.figure(figsize=(14, 8))
        if regression:
            mse, f1, clf_report = visualise_regr_results(model, x_test, y_test, device=device, show_charts=True)
        else:
            mse, f1, clf_report = visualise_clf_results(model, x_test, y_test, device=device, show_charts=True)
        plt.grid()
        plt.title(f'Predictions for split {i}')
        plt.xlabel('Test sample')
        model_type = 'regr' if regression else 'clf'
        plt.savefig(f'results/lstm/predictions_{model_type}_split{i}.png')
        f1_vals.append(f1)
        mse_vals.append(mse)
        clf_reports.append(clf_report)

    return f1_vals, mse_vals, clf_reports, f1_vals_train, mse_vals_train


df = load_data()
f1_vals_regr, mse_vals_regr, clf_reports_regr, f1_vals_train_regr, mse_vals_train_regr = evaluate_mcdlstm(df, True)
f1_vals_clf, mse_vals_clf, clf_reports_clf, f1_vals_train_clf, mse_vals_train_clf = evaluate_mcdlstm(df, False)


if OUT_TO_FILE:
    sys.stdout = open('results/lstm/evaluation.txt', "w")

print(f'Regression without MCD: MSE: {np.mean(mse_vals_train_regr):.3f} ({np.std(mse_vals_train_regr):.3f}), f1: {np.mean(f1_vals_train_regr):.3f}({np.std(f1_vals_train_regr):.3f})')
print(f'Regression with MCD: MSE: {np.mean(mse_vals_regr):.3f} ({np.std(mse_vals_regr):.3f}), f1: {np.mean(f1_vals_regr):.3f}({np.std(f1_vals_regr):.3f})')
print(f'Classifier without MCD: MSE: {np.mean(mse_vals_clf):.3f} ({np.std(mse_vals_clf):.3f}), f1: {np.mean(f1_vals_clf):.3f}({np.std(f1_vals_clf):.3f})')
print(f'Classifier with  MCD: MSE: {np.mean(mse_vals_train_clf):.3f} ({np.std(mse_vals_train_clf):.3f}), f1: {np.mean(f1_vals_train_clf):.3f}({np.std(f1_vals_train_clf):.3f})')

print('\n Classification report for classification model, fold 3')
print(clf_reports_clf[2])
