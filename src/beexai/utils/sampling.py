from typing import Tuple

import numpy as np
import pandas as pd
import torch

from beexai.utils.convert import convert_to_numpy


def stratified_sampling(x, y, n, task) -> Tuple[pd.DataFrame, pd.Series]:
    """Stratified sampling of a dataset

    Args:
        x (pd.DataFrame): features
        y (pd.Series): labels
        n (int): number of samples to take
        task (str): task

    Returns:
        pd.DataFrame: sampled features
        pd.Series: sampled labels
    """
    assert x.shape[0] == y.shape[0], "x and y must have the same number of rows"
    assert task in [
        "classification",
        "regression",
    ], f"task must be in ['classification', 'regression'], found {task}"
    if task == "regression":
        return x[:n], y[:n]
    n_samples = min(n, x.shape[0])
    if n_samples == x.shape[0]:
        return x, y
    label_dist = y.value_counts(normalize=True)
    sample_size = (label_dist * n_samples).astype(int)
    sample_size.loc[sample_size == 0] = 1
    sample_size = sample_size.to_dict()
    x_sample = pd.DataFrame()
    y_sample = pd.DataFrame()
    for index in sample_size:
        sample_index = np.random.choice(y[y == index].index, sample_size[index])
        x_sample = pd.concat([x_sample, x.loc[sample_index]])
        y_sample = pd.concat([y_sample, y.loc[sample_index]])
    return x_sample, y_sample


def get_sample_pred(
    model, x, y, task="regression", for_true=True, threshold=0.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """Get samples of x that have a true (or a false) prediction

    Args:
        model (nn.Module): model to use
        x (pd.DataFrame): features
        y (pd.Series): labels
        task (str, optional): task. Defaults to "regression".
        for_true (bool, optional): True if we want the samples that have
            a true prediction. Defaults to True.
        threshold (float, optional): threshold for the prediction.
            Defaults to 0.5.

    Returns:
        pd.DataFrame: sampled features
        pd.Series: sampled labels
    """
    assert x.shape[0] == y.shape[0], "x and y must have the same number of rows"
    assert task in [
        "classification",
        "regression",
    ], f"task must be in ['classification', 'regression'], found {task}"
    with torch.no_grad():
        y_pred = model.predict(x).squeeze()
    y_pred = convert_to_numpy(y_pred)
    if task == "regression":
        threshold = threshold * np.std(y_pred)
    print(f"Threshold: {threshold}")
    if for_true:
        true_indexes = np.where(np.abs(y_pred - y) <= threshold)[0]
    else:
        true_indexes = np.where(np.abs(y_pred - y) > threshold)[0]
    x_test = x.iloc[true_indexes]
    if x_test.shape[0] == 1:
        x_test = x_test.values.reshape(1, -1)
    else:
        x_test = x_test.values
    assert x_test.shape[0] > 0, "No samples found for the given threshold"
    x_test = pd.DataFrame(x_test, columns=x.columns)
    y_test = y.iloc[true_indexes]
    x_test, y_test = x_test.reset_index(drop=True), y_test.reset_index(drop=True)
    return x_test, y_test
