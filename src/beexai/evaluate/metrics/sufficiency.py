from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Sufficiency(CustomMetric):
    """Implementation of the sufficiency metric.

    Computes the sufficiency of the model by adding the most
    important features one by one and computing the difference
    in prediction with the original input.

    References:
        - `ERASER: A Benchmark to Evaluate Rationalized NLP Models
            <https://arxiv.org/abs/1911.03429>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_sufficiency: computes the sufficiency of the model
        get_mr_list: computes the sufficiency of the model for
            different ratios of features added
    """

    def get_sufficiency(
        self,
        x_in: torch.Tensor,
        feature_by_importance: torch.Tensor,
        keep_ratio: Union[float, list] = 0.3,
        label: Optional[Union[int, list, torch.Tensor]] = None,
        baseline: str = "zero",
    ) -> float:
        """Computes the sufficiency of the model.

        Args:
            x_in (torch.Tensor): input data
            feature_by_importance (torch.Tensor): indexes of most
                important features in descending order
            keep_ratio (float, optional): ratio of features to keep.
                Defaults to 0.3.
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            baseline (str, optional): baseline to use. Defaults to "zero".

        Returns:
            float: sufficiency score
        """
        all_suff = 0
        if isinstance(keep_ratio, float):
            ratios = [keep_ratio]
        else:
            ratios = keep_ratio
        for kp_ratio in ratios:
            self.check_shape(x_in, feature_by_importance)
            pred_allf, max_arg = self.select_output(x_in, label)
            n_feats = x_in.shape[1]
            n_feats_k = int(n_feats * kp_ratio)
            input_kf = self.choose_baseline(x_in, baseline, device=self.device)
            indexes_to_remove = feature_by_importance[:, :n_feats_k]
            r_ind = torch.arange(len(indexes_to_remove))[:, None]
            c_ind = indexes_to_remove
            input_kf[r_ind, c_ind] = x_in[r_ind, c_ind]
            # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
            # for i in range(feature_by_importance.shape[0]):
            #     for j in range(n_feats_k):
            #         index = feature_by_importance[i][j]
            #         input_kf[i][index] = x_in[i][index]
            if label is not None:
                pred_kf = self.get_predlb(input_kf, label)
            else:
                pred_kf = self.get_predlb(input_kf, max_arg)
            diff = pred_allf - pred_kf
            if self.task == "regression":
                diff = torch.abs(diff)
            suff = torch.mean(diff, axis=0).item()
            all_suff += suff
        return all_suff / len(ratios)

    def get_mr_list(
        self,
        n_features: int,
        x_test: torch.Tensor,
        orders: torch.Tensor,
        n_plot: int,
        baseline: str = "zero",
        label: Optional[Union[int, list, torch.Tensor]] = None,
    ) -> List[float]:
        """Compute the sufficiency of the model for different
        ratios of features added.

        Args:
            n_features (int): number of features
            x_test (torch.Tensor): test data
            orders (torch.Tensor): indexes of most important features
                in descending order
            n_plot (int): number of points to plot
            baseline (str, optional): baseline to use. Defaults to "zero".
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.

        Returns:
            list: list of sufficiency scores
        """
        assert 0 < n_plot, "n_plot must be positive"
        suff_list = []
        for i in range(n_plot):
            k_r = i / n_features
            score = self.get_sufficiency(x_test, orders, k_r, label, baseline)
            suff_list.append(score)
        return suff_list


def plot_suff(
    n_features: int,
    suff_list: List[float],
    rand_suff_list: List[float],
    randmodel_suff_list: List[float],
    n_plot: int,
    same_fig: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Plot the sufficiency of the base model, the random
    explainer and the random model for different ratios of features
    added.

    Args:
        n_features (int): number of features
        suff_list (list): sufficiency list for the base model
        rand_suff_list (list): sufficiency list for
            the random explainer
        randmodel_suff_list (list): sufficiency list for
            the random model
        n_plot (int): number of points to plot
        same_fig (bool, optional): whether to plot on the same figure.
            Defaults to False.
        save_path (str, optional): path to save the plot. Defaults to None.
    """
    assert 0 < n_plot, "n_plot must be positive"
    x_axis = np.arange(n_plot) / n_features
    if not same_fig:
        fig = plt.figure()
        subplot_size = 110
        if rand_suff_list:
            subplot_size += 100
        if randmodel_suff_list:
            subplot_size += 100
        ax1 = fig.add_subplot(subplot_size + 1)
        ax1.plot(x_axis, suff_list, label="Sufficiency")
        ax1.legend()
        if rand_suff_list:
            ax2 = fig.add_subplot(subplot_size + 2)
            ax2.plot(x_axis, rand_suff_list, label="Random Sufficiency")
            ax2.legend()
        if randmodel_suff_list:
            ax3 = fig.add_subplot(subplot_size + 3)
            ax3.plot(x_axis, randmodel_suff_list, label="Random Model Sufficiency")
            ax3.legend()
        plt.xlabel("Ratio of features kept")
        plt.ylabel("Mean absolute difference")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    else:
        plt.plot(x_axis, suff_list, label="Sufficiency")
        plt.plot(x_axis, rand_suff_list, label="Random Sufficiency")
        plt.plot(x_axis, randmodel_suff_list, label="Random Model Sufficiency")
        plt.xlabel("Ratio of features kept")
        plt.ylabel("Mean probability difference")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


@time_function
def compute_suff(
    model: Callable,
    ref_model: Callable,
    task: str,
    x_test: torch.Tensor,
    orders: torch.Tensor,
    rand_orders: torch.Tensor,
    randmodel_orders: torch.Tensor,
    n_plot: int,
    feature_ratio: Union[float, list],
    label: Union[int, list, torch.Tensor],
    metrics: dict,
    baseline: str = "zero",
    print_plot: bool = False,
    device: str = "cpu",
) -> dict:
    """Computes the sufficiency of the base model, the random
    explainer and the random model.

    Args:
        model (callable): base model
        ref_model (callable): reference model (random model)
        task (str): task to perform
        x_test (torch.Tensor): test data
        orders (torch.Tensor): indexes of most important features
            in descending order for the base model
        rand_orders (torch.Tensor): indexes of most important features
            in descending order for the random explainer
        randmodel_orders (torch.Tensor): indexes of most important features
            in descending order for the random model
        n_plot (int): number of points to plot
        feature_ratio (float, list): ratio of features to keep. A list of
            labels for each instance can be provided.
        label (int, list, np.ndarray, torch.Tensor): label(s) of interest
        metrics (dict): dictionary of metrics
        baseline (str, optional): baseline to use. Defaults to "zero".
        print_plot (str, optional): whether to print the plot.
            Defaults to False.
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        dict: dict of metrics
    """
    n_features = x_test.shape[1]
    suff = Sufficiency(model, task, device)
    use_ref = ref_model is not None
    use_random = rand_orders is not None
    if use_ref:
        randmodel_suff = Sufficiency(ref_model, task, device)
    else:
        randmodel_suff = None
    if print_plot:
        suff_list = suff.get_mr_list(
            n_features, x_test, orders, n_plot, baseline, label
        )
        if use_random:
            rand_suff_list = suff.get_mr_list(
                n_features, x_test, rand_orders, n_plot, baseline, label
            )
        else:
            rand_suff_list = []
        if use_ref:
            randmodel_suff_list = randmodel_suff.get_mr_list(
                n_features, x_test, randmodel_orders, n_plot, baseline, label
            )
        else:
            randmodel_suff_list = []
        plot_suff(n_features, suff_list, rand_suff_list, randmodel_suff_list, n_plot)
    suff_score = suff.get_sufficiency(x_test, orders, feature_ratio, label, baseline)
    metrics["Sufficiency"]["original"] = suff_score
    if use_random:
        rand_suff_score = suff.get_sufficiency(
            x_test, rand_orders, feature_ratio, label, baseline
        )
        metrics["Sufficiency"]["random"] = rand_suff_score
    if use_ref:
        randmodel_suff_score = randmodel_suff.get_sufficiency(
            x_test, randmodel_orders, feature_ratio, label, baseline
        )
        metrics["Sufficiency"]["randommodel"] = randmodel_suff_score
    return metrics
