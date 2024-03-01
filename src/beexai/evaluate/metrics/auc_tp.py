from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.training.train import Trainer
from beexai.utils.convert import convert_to_tensor
from beexai.utils.time_seed import time_function


class AucTp(CustomMetric):
    """Implementation of the AUC-TP metric.

    Computes the AUC of the curve with x-axis being the number
    of features removed and y-axis being the performance metric
    between the prediction with all features and the prediction
    with the x most important features removed.

    References:
        - `A Diagnostic Study of Explainability Techniques for
        Text Classification <https://arxiv.org/abs/2009.13295>`

    Attributes:
        model (Trainer): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_auctp: computes the AUC-TP metric
    """

    def get_auctp(
        self,
        x_in: torch.Tensor,
        feature_by_importance: torch.Tensor,
        metric: Callable,
        baseline: str = "zero",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the AUC-TP metric.

        Args:
            x_in (torch.Tensor): x_in data
            feature_by_importance (torch.Tensor): indexes of most
                important features in descending order
            metric (callable): metric to use to compute the difference
                between the prediction with all features and
                the prediction with the most important features removed
            baseline (str, optional): baseline to use. Defaults to "zero"

        Returns:
            torch.Tensor: array of AUC-TP scores for each feature
            torch.Tensor: AUC-TP score
        """
        self.check_shape(x_in, feature_by_importance)
        auc_array = torch.zeros(feature_by_importance.shape[1], device=self.device)
        with torch.no_grad():
            pred = self.model.predict(x_in)
        pred = convert_to_tensor(pred, self.device)
        baseline_values = self.choose_baseline(x_in, baseline, device=self.device)
        for i in range(feature_by_importance.shape[1]):
            input_rmf = baseline_values.clone()
            indexes_to_keep = feature_by_importance[:, i:]
            r_ind = torch.arange(len(indexes_to_keep))[:, None]
            c_ind = indexes_to_keep
            input_rmf[r_ind, c_ind] = x_in[r_ind, c_ind]
            # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
            # input_rmf = torch.zeros(x_in.shape).to(self.device)
            # for j in range(feat_imp.shape[0]):
            #     input_rmf[j] = baseline_values[j]
            #     for k in range(num_features_to_remove,feat_imp.shape[1]):
            #         index = feat_imp[j][k]
            #         input_rmf[j][index] = input_tensor[j][index]
            with torch.no_grad():
                pred_rmf = self.model.predict(input_rmf)
            pred_rmf = convert_to_tensor(pred_rmf, self.device)
            auc_array[i] = metric(pred, pred_rmf)
        # Regression: AUC((1/MSE)*ind(dmse/dratio > 0))
        if self.task == "regression":
            auc_array = auc_array + 0.01 * torch.mean(auc_array)
            x_r = torch.arange(
                1, feature_by_importance.shape[1] + 1, device=self.device
            )
            dratio = 1 - 1 / (x_r + 1e-8)
            dmse = torch.zeros(auc_array.shape, device=self.device)
            for i in range(1, len(auc_array) - 1):
                dmse[i] = (auc_array[i + 1] - auc_array[i - 1]) / (
                    dratio[i + 1] - dratio[i - 1]
                )
            ind = (dmse / dratio > 0).int()
            auc_array = (1 / auc_array) * ind
            max_auc = torch.max(auc_array)
            min_auc = torch.min(auc_array)
            auc_array = (auc_array - min_auc) / (max_auc - min_auc + 1e-8)
            auctp = torch.trapz(auc_array) / x_in.shape[1]
        else:
            auctp = torch.trapz(auc_array) / x_in.shape[1]
        return auc_array, auctp


def plot_metric(
    p_curve: torch.Tensor,
    rand_p_curve: torch.Tensor,
    randmodel_p_curve: torch.Tensor,
    same_fig: bool = False,
) -> None:
    """Plots the performance curve for the original model,
    the random model and the random baseline.

    Args:
        p_curve (torch.Tensor): performance curve for the base model
            and base explanation method
        rand_p_curve (torch.Tensor): performance curve for the base model
            and random explanation method
        randmodel_p_curve (torch.Tensor): performance curve for
            the random model and base explanation method
        same_fig (bool, optional): whether to plot all curves on the
            same figure. Defaults to False
    """
    if not same_fig:
        fig = plt.figure()
        subplot_size = 110
        if rand_p_curve:
            subplot_size += 100
        if randmodel_p_curve:
            subplot_size += 100
        ax1 = fig.add_subplot(subplot_size + 1)
        ax1.plot(p_curve, label="AUC")
        ax1.legend()
        if rand_p_curve:
            ax2 = fig.add_subplot(subplot_size + 2)
            ax2.plot(rand_p_curve, label="Random AUC")
            ax2.legend()
        if randmodel_p_curve:
            ax3 = fig.add_subplot(subplot_size + 3)
            ax3.plot(randmodel_p_curve, label="Random Model AUC")
            ax3.legend()
        plt.xlabel("Ratio of features removed")
        plt.ylabel("Metric")
        plt.show()
    else:
        plt.plot(p_curve, label="Original")
        plt.plot(rand_p_curve, label="Random")
        plt.plot(randmodel_p_curve, label="Random Trainer")
        plt.xlabel("Ratio of features removed")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()


@time_function
def compute_auc(
    model: Trainer,
    rand_model: Trainer,
    task: str,
    x_test: torch.Tensor,
    ord_feat: torch.Tensor,
    rand_ord_feat: torch.Tensor,
    randmodel_ord_feat: torch.Tensor,
    metrics: dict,
    baseline: str = "zero",
    auc_metric: str = "mse",
    print_plot: bool = False,
    device: str = "cpu",
) -> dict:
    """Computes the AUC-TP metric for the base model, the random model
    and the random baseline. Returns the dict of metrics with the
    computed AUC-TP scores appended.

    Args:
        model (Trainer): model to explain
        rand_model (callable): reference model to compare to (random model)
        task (str): task to perform
        x_test (torch.Tensor): test set
        ord_feat (torch.Tensor): indexes of most important features
            in descending order for the base model
        rand_ord_feat (torch.Tensor): indexes of most important features
            in descending order for the random explanation method
        randmodel_ord_feat (torch.Tensor): indexes of most important
            features in descending order for the random model
        metrics (dict): dictionary of metrics
        baseline (str, optional): baseline to use. Defaults to "zero"
        auc_metric (str, optional): performance metric to use.
            Defaults to "mse"
        print_plot (bool, optional): whether to print the plot.
            Defaults to False
        device (str, optional): device to use. Defaults to "cpu"

    Raises:
        ValueError: auc_metric must be in ['mse','accuracy']

    Returns:
        dict: dict of metrics with the computed AUC-TP scores appended
    """
    use_ref = rand_model is not None
    use_random = rand_ord_feat is not None
    if auc_metric == "mse":

        def mse(y_true, y_pred):
            return torch.mean((y_true - y_pred) ** 2, dtype=torch.float32)

        metric = mse
    elif auc_metric == "accuracy":

        def accuracy(y_true, y_pred):
            return torch.mean(y_true == y_pred, dtype=torch.float32)

        metric = accuracy
    else:
        raise ValueError("auc_metric must be in ['mse', 'accuracy']")
    auctp = AucTp(model, task, device)
    p_curve, auctp_score = auctp.get_auctp(x_test, ord_feat, metric, baseline)
    p_curve = p_curve.detach().cpu().numpy()
    if use_random:
        rand_p_curve, rand_auctp_score = auctp.get_auctp(
            x_test, rand_ord_feat, metric, baseline
        )
        rand_p_curve = list(rand_p_curve.detach().cpu().numpy())
    else:
        rand_p_curve = []
        rand_auctp_score = None
    if use_ref:
        randmodel_auctp = AucTp(rand_model, task, device)
        randmodel_p_curve, randmodel_auctp_score = randmodel_auctp.get_auctp(
            x_test, randmodel_ord_feat, metric, baseline
        )
        randmodel_p_curve = list(randmodel_p_curve.detach().cpu().numpy())
    else:
        randmodel_p_curve = []
        randmodel_auctp_score = None
    if print_plot:
        plot_metric(p_curve, rand_p_curve, randmodel_p_curve)
    metrics["AUC_TP"]["original"] = auctp_score.item()
    if use_random:
        metrics["AUC_TP"]["random"] = rand_auctp_score.item()
    if use_ref:
        metrics["AUC_TP"]["random_model"] = randmodel_auctp_score.item()
    return metrics
