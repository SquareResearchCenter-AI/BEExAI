from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Comprehensiveness(CustomMetric):
    """Implementation of the comprehensiveness metric.

    Computes the comprehensiveness of the model by removing
    the most important features one by one and
    computing the difference in prediction with the original input.

    References:
        - `ERASER: A Benchmark to Evaluate Rationalized NLP Models
        <https://arxiv.org/abs/1911.03429>`

    Attributes:
        model (Callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_comp: computes the comprehensiveness of the model
        get_mr_list: computes the comprehensiveness of the model
            for different ratios of features removed
    """

    def get_comp(
        self,
        x_in: torch.Tensor,
        feature_by_importance: torch.Tensor,
        removal_ratio: Union[float, list] = 0.3,
        label: Optional[Union[int, list, np.ndarray, torch.Tensor]] = None,
        baseline: str = "zero",
    ) -> float:
        """Computes the comprehensiveness of the model.

        Args:
            x_in (torch.Tensor): input data
            feature_by_importance (torch.Tensor): indexes of
                most important features in descending order
            removal_ratio (float, list): ratio of features to remove.
                If a list is provided, the function will compute the
                average comprehensiveness over the list of ratios.
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            baseline (str, optional): baseline to use. Defaults to "zero"

        Returns:
            float: comprehensiveness score
        """
        all_comp = 0
        if isinstance(removal_ratio, float):
            ratios = [removal_ratio]
        else:
            ratios = removal_ratio
        for rm_ratio in ratios:
            self.check_shape(x_in, feature_by_importance)
            pred_allf, max_arg = self.select_output(x_in, label)
            n_feats = x_in.shape[1]
            n_feats_rm = int(n_feats * rm_ratio)
            input_rmf = self.choose_baseline(x_in, baseline, device=self.device)
            indexes_to_keep = feature_by_importance[:, n_feats_rm:]
            r_ind = torch.arange(len(indexes_to_keep))[:, None]
            c_ind = indexes_to_keep
            input_rmf[r_ind, c_ind] = x_in[r_ind, c_ind]
            # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
            # for i in range(feat_imp.shape[0]):
            #     input_rmf[i] = baseline_values[i]
            #     for j in range(n_feats_rm,feat_imp.shape[1]):
            #         index = feat_imp[i][j]
            #         input_rmf[i][index] = X[i][index]
            if label is not None:
                pred_rmf = self.get_predlb(input_rmf, label)
            else:
                pred_rmf = self.get_predlb(input_rmf, max_arg)
            diff = pred_allf - pred_rmf
            if self.task == "regression":
                diff = torch.abs(diff)
            comp = torch.mean(diff, axis=0).item()
            all_comp += comp
        return all_comp / len(ratios)

    def get_mr_list(
        self,
        n_features: int,
        x_test: torch.Tensor,
        orders: torch.Tensor,
        n_plot: int,
        baseline: str = "zero",
        label: Optional[Union[int, list, np.ndarray, torch.Tensor]] = None,
    ) -> List[float]:
        """Compute the comprehensiveness of the model for different
        ratios of features removed.

        Args:
            n_features (int): number of features
            x_test (torch.Tensor): test data
            orders (torch.Tensor): indexes of most important features
                in descending order
            n_plot (int): number of points to plot
            baseline (str, optional): baseline to use. Defaults to "zero".
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.

        Returns:
            list: list of comprehensiveness scores
        """
        assert 0 <= n_plot, "n_plot must be positive"
        comp_list = []
        for i in range(n_plot):
            rm_r = i / n_features
            comp = self.get_comp(
                x_in=x_test,
                feature_by_importance=orders,
                removal_ratio=rm_r,
                label=label,
                baseline=baseline,
            )
            comp_list.append(comp)
        return comp_list


def plot_comp(
    n_features: int,
    comp_list: List[float],
    rand_comp_list: List[float],
    randmodel_comp_list: List[float],
    n_plot: int,
    same_fig: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Plot the comprehensiveness of the base model, the random
    explainer and the random model for different ratios of features
    removed.

    Args:
        n_features (int): number of features
        comp_list (list): comprehensiveness list for the base model
        rand_comp_list (list): comprehensiveness list for
            the random explainer
        randmodel_comp_list (list): comprehensiveness list for
            the random model
        n_plot (int): number of points to plot
        same_fig (bool, optional): whether to plot on the same figure.
            Defaults to False.
        save_path (str, optional): path to save the plot. Defaults to None.
    """
    assert 0 <= n_plot, "n_plot must be positive"
    x_axis = np.arange(n_plot) / n_features
    if not same_fig:
        fig = plt.figure()
        subplot_size = 110
        if rand_comp_list:
            subplot_size += 100
        if randmodel_comp_list:
            subplot_size += 100
        ax1 = fig.add_subplot(subplot_size + 1)
        ax1.plot(x_axis, comp_list, label="Comprehensiveness")
        ax1.legend()
        if rand_comp_list:
            ax2 = fig.add_subplot(subplot_size + 2)
            ax2.plot(x_axis, rand_comp_list, label="Random Comprehensiveness")
            ax2.legend()
        if randmodel_comp_list:
            ax3 = fig.add_subplot(subplot_size + 3)
            ax3.plot(
                x_axis, randmodel_comp_list, label="Random Model Comprehensiveness"
            )
            ax3.legend()
        plt.xlabel("Ratio of features removed")
        plt.ylabel("Mean absolute difference")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    else:
        plt.plot(x_axis, comp_list, label="Comprehensiveness")
        plt.plot(x_axis, rand_comp_list, label="Random Comprehensiveness")
        plt.plot(x_axis, randmodel_comp_list, label="Random Model Comprehensiveness")
        plt.xlabel("Ratio of features removed")
        plt.ylabel("Mean probability difference")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


@time_function
def compute_comp(
    model: Callable,
    rand_model: Callable,
    task: str,
    x_test: torch.Tensor,
    ord_feat: torch.Tensor,
    rand_ord_feat: torch.Tensor,
    randmodel_ord_feat: torch.Tensor,
    n_plot: int,
    removal_ratio: Union[float, list],
    label: Union[int, list, np.ndarray, torch.Tensor],
    metrics: dict,
    baseline: str = "zero",
    print_plot: bool = False,
    device: str = "cpu",
) -> dict:
    """Computes the comprehensiveness of the base model, the random
    explainer and the random model.

    Args:
        model (Callable): model to explain
        rand_model (Callable): reference model
        task (str): task to perform
        x_test (torch.Tensor): test data
        ord_feat (torch.Tensor): indexes of most important features
            in descending order for the base model
        rand_ord_feat (torch.Tensor): indexes of most important features
            in descending order for the random explainer
        randmodel_ord_feat (torch.Tensor): indexes of most important features
            in descending order for the random model
        n_plot (int): number of points to plot
        removal_ratio (float, list): ratio of features to remove. If a list is
            provided, the function will compute the average comprehensiveness
            over the list of ratios.
        label (Union[int, list, np.ndarray, torch.Tensor]): label(s) of interest
        metrics (dict): dictionary of metrics
        baseline (str, optional): baseline to use. Defaults to "zero".
        print_plot (bool, optional): whether to display the plot.
            Defaults to False.
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        dict: dict of metrics
    """
    n_features = x_test.shape[1]
    comp = Comprehensiveness(model, task, device)
    use_ref = rand_model is not None
    use_random = rand_ord_feat is not None
    if use_ref:
        randmodel_comp = Comprehensiveness(rand_model, task, device)
    else:
        randmodel_comp = None
    if print_plot:
        comp_list = comp.get_mr_list(
            n_features, x_test, ord_feat, n_plot, baseline, label
        )
        if use_random:
            rand_comp_list = comp.get_mr_list(
                n_features, x_test, rand_ord_feat, n_plot, baseline, label
            )
        else:
            rand_comp_list = []
        if use_ref:
            randmodel_comp_list = randmodel_comp.get_mr_list(
                n_features, x_test, randmodel_ord_feat, n_plot, baseline, label
            )
        else:
            randmodel_comp_list = []
        plot_comp(n_features, comp_list, rand_comp_list, randmodel_comp_list, n_plot)
    comp_score = comp.get_comp(x_test, ord_feat, removal_ratio, label, baseline)
    metrics["Comprehensiveness"]["original"] = comp_score
    if use_random:
        rand_comp_score = comp.get_comp(
            x_test, rand_ord_feat, removal_ratio, label, baseline
        )
        metrics["Comprehensiveness"]["random"] = rand_comp_score
    if use_ref:
        randmodel_comp_score = randmodel_comp.get_comp(
            x_test, randmodel_ord_feat, removal_ratio, label, baseline
        )
        metrics["Comprehensiveness"]["random model"] = randmodel_comp_score
    return metrics
