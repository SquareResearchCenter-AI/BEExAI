"""Radar plot for explanation metrics"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

linestyles = [
    ("solid", "solid"),
    ("dotted", "dotted"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("densely dotted", (0, (1, 1))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]

color_palette = [
    "#66C5CC",
    "#F6CF71",
    "#F89C74",
    "#87C55F",
    "#DCB0F2",
    "#9EB9F3",
]

metrics_plot_rename = {
    "FaithCorr_1-": "Faith ↗",
    "Sensitivity_0+": "Sens ↘",
    "Infidelity_0+": "Inf ↘",
    "Comprehensiveness_1-": "Compr ↗",
    "Sufficiency_0+": "Suff ↘",
    "AUC_TP_0+": "AUC-TP ↘",
    "Monotonicity_1-": "Mono ↗",
    "Complexity_0+": "Compl ↘",
    "Sparseness_1-": "Spar ↗",
}

metrics_range = {
    "FaithCorr_1-": (0.0, 1.0),
    "Sensitivity_0+": (1.0, 0.0),
    "Infidelity_0+": (1.0, 0.0),
    "Comprehensiveness_1-": (0.0, 1.0),
    "Sufficiency_0+": (1.0, 0.0),
    "AUC_TP_0+": (1.0, 0.0),
    "Complexity_0+": (1.0, 0.0),
    "Sparseness_1-": (0.0, 1.0),
    "Monotonicity_1-": (0.0, 1.0),
}


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1), f"d={d}, y1={y1}, y2={y2}"
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
    return sdata


class ComplexRadar:
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360.0 / len(variables))
        axes = [
            fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True, label=f"axes{i}")
            for i in range(len(variables))
        ]

        for label, i in zip(axes[0].get_xticklabels(), range(0, len(angles))):
            angle_rad = angles[i]
            if angle_rad <= pi / 2:
                ha = "left"
                va = "bottom"

            elif pi / 2 < angle_rad <= pi:
                ha = "right"
                va = "bottom"

            elif pi < angle_rad <= (3 * pi / 2):
                ha = "right"
                va = "top"

            else:
                ha = "right"
                va = "bottom"

            label.set_verticalalignment(va)
            label.set_horizontalalignment(ha)

        _, text = axes[0].set_thetagrids(angles, labels=variables)
        [txt.set_rotation(angle - 90) for txt, angle in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = [f"{round(x,2)}" for x in grid]
            gridlabel[0] = ""
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            ax.set_ylim(*ranges[i])
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def plot_multiple(self, data, methods, *args, **kw):
        for i, d in enumerate(data):
            _, marker = linestyles[i]
            color = color_palette[i]
            sdata = _scale_data(d, self.ranges)
            self.ax.plot(
                self.angle,
                np.r_[sdata, sdata[0]],
                linestyle=marker,
                linewidth=3,
                c=color,
                *args,
                **kw,
            )
        self.ax.tick_params(axis="both", which="major", pad=18)
        self.ax.legend(methods, loc="lower right", bbox_to_anchor=(1.4, 0.1))

    def fill_multiple(self, data, *args, **kw):
        for i, d in enumerate(data):
            color = color_palette[i]
            sdata = _scale_data(d, self.ranges)
            self.ax.fill(self.angle, np.r_[sdata, sdata[0]], color=color, *args, **kw)


def get_dec_exponent(x):
    """Get the decimal exponent of a number x"""
    x_str = "{:e}".format(x)
    x_str = x_str.split("e")
    return float(x_str[0]), int(x_str[1])


def plot_metric(
    df_path,
    metrics_plot=[
        "FaithCorr_1-",
        "Sensitivity_0+",
        "Infidelity_0+",
        "Comprehensiveness_1-",
        "Sufficiency_0+",
        "AUC_TP_0+",
        "Monotonicity_1-",
        "Complexity_0+",
        "Sparseness_1-",
    ],
    methods_plot=[
        "Lime",
        "ShapleyValueSampling",
        "KernelShap",
        "DeepLift",
        "IntegratedGradients",
        "Saliency",
    ],
    plot_nn=True,
    save_path=None,
    alpha=0.2,
) -> None:
    """Plot the radar chart for the metrics.

    Args:
        df_path: path for the metrics dataframe
        metrics_plot: list of metrics to plot
        methods_plot: list of methods to plot
        plot_nn: whether to plot metric for Neural Network
        save_path: path to save the plot
        alpha: transparency of the plot
    """
    metric_df = pd.read_csv(df_path)
    metric_df.index = metric_df["metrics"]
    metric_df = metric_df.loc[methods_plot]
    metrics_plot_copy = metrics_plot.copy()

    values = []
    for i in range(len(metrics_plot)):
        col_name = metrics_plot[i]
        if not plot_nn:
            col_name = col_name + ".1"
        metrics_col = metric_df[metrics_plot[i]].dropna().tolist()
        metrics_col = [float(i) for i in metrics_col]
        values.append(metrics_col)
    values = np.array(values)

    ranges = [metrics_range[key] for key in metrics_plot]
    metrics_plot_re = [metrics_plot_rename[metric] for metric in metrics_plot]
    for i, r in enumerate(ranges):
        mean_std = np.std(values[i])
        if mean_std < 0.005:
            dec_exp = get_dec_exponent(mean_std)
            values[i] = [x * 10 ** (-dec_exp[1] - 1) for x in values[i]]
            ranges[i] = (r[0] * 10 ** (-dec_exp[1] - 1), r[1] * 10 ** (-dec_exp[1]))
            metrics_plot_copy[i] = metrics_plot_re[i] + " (10e" + str(dec_exp[1]) + ")"
        else:
            metrics_plot_copy[i] = metrics_plot_re[i]
        max_val = max(values[i])
        min_val = min(values[i])
        margin = 2.2 * (max_val - min_val)
        if values.shape[1] > 1:
            ranges[i] = (
                (max_val - margin, max_val)
                if r[0] < r[1]
                else (min_val + margin, min_val)
            )
        else:
            ranges[i] = (
                (max_val - 0.1, max_val) if r[0] < r[1] else (min_val + 0.1, min_val)
            )
    fig1 = plt.figure(figsize=(10, 10))
    radar = ComplexRadar(fig1, metrics_plot_copy, ranges)
    radar.plot_multiple(values.transpose(), methods_plot)
    radar.fill_multiple(values.transpose(), alpha=alpha)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
