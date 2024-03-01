"""Plotting functions for feature attributions"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def bar_plot(
    attributions: np.ndarray, feature_names: Optional[list] = None, mean: bool = False
) -> None:
    """Plot a bar plot for the given attributions

    Args:
        attributions (np.ndarray): attributions
        feature_names (Optional[list], optional): name of the features. Defaults to None.
           If None is provided, they will be named as "Feature i".
        mean (bool, optional): whether to average the attributions. Defaults to False.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(attributions.shape[0])]
    if mean or attributions.ndim > 1 and attributions.shape[0] > 1:
        attributions = attributions.mean(axis=0)
    _, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(feature_names, attributions)
    ax.set_xlabel("Attributions")
    ax.set_ylabel("Features")
    ax.set_title("Feature attributions")
    ax.bar_label(bars, fmt="%.5f")
    for i, plot_bar in enumerate(bars):
        if attributions[i] < 0:
            plot_bar.set_color("r")
    plt.margins(0.1)
    plt.show()


def plot_waterfall(
    attributions: np.ndarray,
    feature_names: Optional[list] = None,
    mean: bool = False,
) -> None:
    """Plot a waterfall plot for the given attributions

    Args:
        attributions (np.ndarray): attributions
        feature_names (Optional[list], optional): name of the features. Defaults to None.
           If None is provided, they will be named as "Feature i".
        mean (bool, optional): whether to average the attributions. Defaults to False.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(attributions.shape[0])]
    if mean or attributions.ndim > 1 and attributions.shape[0] > 1:
        attributions = attributions.mean(axis=0)
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative" for _ in feature_names],
            x=attributions,
            textposition="outside",
            text=[
                f"{name}: {value:.5f}"
                for name, value in zip(feature_names, attributions)
            ],
            y=feature_names,
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}},
            connector={"line": {"color": "black"}},
        )
    )
    fig.update_layout(title="Feature attributions")
    fig.show()


def plot_swarm(
    x_in: pd.DataFrame, attributions: np.ndarray, feature_names: Optional[list] = None
) -> None:
    """Plot a swarm plot for the given attributions
    Args:
        x_in (pd.DataFrame): input data
        attributions (np.ndarray): attributions
        feature_names (Optional[list], optional): name of the features. Defaults to None.
           If None is provided, they will be named as "Feature i".
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(attributions.shape[0])]
    fig, axs = plt.subplots(len(feature_names), 1, figsize=(10, 10))
    norm = plt.Normalize(
        vmin=min(x_in.values.flatten()), vmax=max(x_in.values.flatten())
    )
    for i, feature in enumerate(feature_names):
        x_in[feature] = x_in[feature].astype(float)

        ax = axs[i]
        a = attributions[:, i]
        c = x_in[feature].values
        key = [feature] * len(a)
        df = pd.DataFrame({"key": key, "a": a, "c": c})

        cmap = sns.color_palette("coolwarm", as_cmap=True)
        colors = {}
        for cval in df["c"].unique():
            colors.update({cval: cmap((cval))})

        sns.stripplot(
            x="a",
            y="key",
            hue="c",
            data=df,
            palette=colors,
            ax=ax,
            orient="h",
            jitter=0.2,
        )
        ax.legend_.remove()
        ax.set_ylabel("")
        ax.set_xlabel("Attribution values")

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), ax=axs, orientation="vertical", norm=norm
    )
    plt.show()
