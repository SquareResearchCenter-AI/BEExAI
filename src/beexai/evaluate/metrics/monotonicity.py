from typing import Callable, Optional, Union

import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Monotonicity(CustomMetric):
    """Implementation of the monotonicity metric.

    Computes the monotonicity of the model by adding the least
    important features one by one.
    If pred(i+1) - pred(i) <= pred(i+2) - pred(i+1)
    for i in range(0, n_features-2) then assign to 1,
    else 0 then take the mean for all features.

    References:
        - `Synthetic Benchmarks for Scientific Research in Explainable
        Machine Learning <https://arxiv.org/abs/2106.12543>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_mono: computes the monotonicity of the model
    """

    def get_mono(
        self,
        x_in: torch.Tensor,
        feature_by_importance: torch.Tensor,
        label: Optional[Union[int, list, torch.Tensor, int]] = None,
        baseline: str = "zero",
    ) -> float:
        """Computes the monotonicity of the model.

        Args:
            x_in (pd.DataFrame): input data
            feature_by_importance (torch.Tensor): indexes of most
                important features in descending order
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            baseline (str, optional): baseline to use. Defaults to "zero".

        Returns:
            float: monotonicity score
        """
        self.check_shape(x_in, feature_by_importance)
        pred_full, max_arg = self.select_output(x_in, label)
        baseline_values = self.choose_baseline(x_in, baseline, device=self.device)
        pred_0, _ = self.select_output(baseline_values, label)
        pred_t = torch.zeros((x_in.shape[1] + 1, x_in.shape[0]), device=self.device)
        pred_t[0] = pred_0.squeeze()
        for i in range(x_in.shape[1]):
            num_features_to_keep = i + 1
            input_kf = baseline_values.clone()
            indexes_to_remove = feature_by_importance[:, :num_features_to_keep]
            r_ind = torch.arange(len(indexes_to_remove))[:, None]
            c_ind = indexes_to_remove
            input_kf[r_ind, c_ind] = x_in[r_ind, c_ind]
            # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
            # for j in range(feat_imp.shape[0]):
            #     input_with_kept_features[j] = baseline_values[j]
            #     for k in range(num_features_to_keep):
            #         index = feat_imp[j][k]
            #         if index != 0:
            #             input_with_kept_features[j][index] = X[j][index]
            if label is not None:
                pred_kf = self.get_predlb(input_kf, label)
            else:
                pred_kf = self.get_predlb(input_kf, max_arg)
            if self.task == "regression":
                pred_t[i + 1] = torch.abs(pred_kf.squeeze() - pred_full)
            else:
                pred_t[i + 1] = pred_full - pred_kf.squeeze()
        diff = torch.diff(torch.mean(pred_t, axis=1), dim=0)
        improve_count = torch.mean(
            torch.where(diff[0:-1] <= diff[1:], 1, 0), axis=0, dtype=torch.float32
        ).squeeze()
        return improve_count.item()


@time_function
def compute_mono(
    model: Callable,
    rand_model: Callable,
    task: str,
    x_test: torch.Tensor,
    ord_feat: torch.Tensor,
    rand_ord_feat: torch.Tensor,
    randmodel_ord_feat: torch.Tensor,
    label: Union[int, list, torch.Tensor],
    metrics: dict,
    baseline: str = "zero",
    device: str = "cpu",
) -> dict:
    """Compute the monotonicity metric.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task of the model
        x_test (torch.Tensor): test data
        ord_feat (torch.Tensor): indexes of most important features
            in descending order for the base model
        rand_ord_feat (torch.Tensor): indexes of most important features
            in descending order for the random explainer
        randmodel_ord_feat (torch.Tensor): indexes of most important
            features in descending order for the random model
        label (int, list, np.ndarray, torch.Tensor): label(s) of interest
        metrics (dict): dictionary of metrics
        baseline (str, optional): baseline to use. Defaults to "zero".
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        dict: dict of metrics
    """
    mono = Monotonicity(model, task, device)
    mono_score = mono.get_mono(x_test, ord_feat, label, baseline)
    metrics["Monotonicity"]["original"] = mono_score

    if rand_ord_feat is not None:
        rand_mono_score = mono.get_mono(x_test, rand_ord_feat, label, baseline)
        metrics["Monotonicity"]["random"] = rand_mono_score

    if rand_model is not None:
        randmodel_mono = Monotonicity(rand_model, task, device)
        randmodel_mono_score = randmodel_mono.get_mono(
            x_test, randmodel_ord_feat, label, baseline
        )
        metrics["Monotonicity"]["random model"] = randmodel_mono_score
    return metrics
