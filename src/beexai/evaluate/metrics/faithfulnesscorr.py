from typing import Callable, Optional, Union

import numpy as np
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class FaithfulnessCorrelation(CustomMetric):
    """Implementation of the faithfulness correlation metric.

    Computes the faithfulness of the model by removing a fixed
    number of features and compute the Pearson correlation between
    the summed attributions and the difference in prediction
    with the original input.

    References:
        - `Synthetic Benchmarks for Scientific Research in Explainable
        Machine Learning <https://arxiv.org/abs/2106.12543>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_faithfulness: computes the faithfulness of the model
    """

    def get_faithfulness(
        self,
        x_in: torch.Tensor,
        attributions: torch.Tensor,
        n_features_subset: int,
        label: Optional[Union[int, list, np.ndarray, torch.Tensor]] = None,
        n_repeats: int = 20,
        baseline: str = "zero",
    ) -> float:
        """Computes the faithfulness of the model.

        Args:
            x_in (torch.Tensor): input data
            attributions (torch.Tensor): attributions for each instance
            n_features_subset (int): number of features to remove
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            n_repeats (int, optional): number of times to repeat the
                sampling. Defaults to 20.
            baseline (str, optional): baseline to use. Defaults to "zero".

        Returns:
            float: faithfulness score
        """
        self.check_shape(x_in, attributions)
        n_samples, n_features = x_in.shape
        n_features_subset = max(n_features_subset, 1)
        assert n_features_subset > 0, "n_features_subset must be > 0"
        assert n_repeats > 0, "n_repeats must be > 0"
        pred, max_arg = self.select_output(x_in, label=label)
        feature_subsets = torch.randint(n_features, size=(n_repeats, n_features_subset))
        baseline_values = self.choose_baseline(x_in, baseline, device=self.device)
        deltas = torch.zeros((n_repeats, n_samples), device=self.device)
        sums = torch.zeros((n_repeats, n_samples), device=self.device)
        for j in range(n_repeats):
            feature_subset = feature_subsets[j]
            mask = torch.ones(x_in.shape, device=self.device)
            r_ind = torch.arange(n_samples)[:, None]
            c_ind = feature_subset
            mask[r_ind, c_ind] = baseline_values[r_ind, c_ind]
            x_pert = torch.multiply(x_in, mask)
            pred_new = self.get_predlb(x_pert, max_arg)
            deltas[j] = (pred - pred_new).squeeze()
            if self.task == "regression":
                deltas[j] = torch.abs(deltas[j])
            sums[j] = torch.mean(attributions[r_ind, c_ind], axis=1).squeeze()
        deltas_flat = deltas.flatten().detach().cpu().numpy()
        sums_flat = sums.flatten().detach().cpu().numpy()
        faithfulness = np.corrcoef(deltas_flat, sums_flat)[0, 1]
        # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
        # for i in range(X.shape[0]):
        #     deltas = np.zeros(n_repeats)
        #     sums = np.zeros(n_repeats)
        #     for j in range(n_repeats):
        #         feature_subset = feature_subsets[j]
        #         mask = torch.ones(X.shape[1:],device=self.device)
        #         mask[feature_subset] = baseline_values[i][feature_subset]
        #         x_pert = torch.multiply(X[i],mask).unsqueeze(0)
        #         pred_new = self.get_predlb(x_pert,max_arg)
        #         deltas[j] = abs(pred[i]-pred_new).squeeze()
        #         sums[j] = np.mean(attributions[i][feature_subset]).squeeze()
        #     faithfulness = np.corrcoef(deltas,sums)[0,1]
        #     if np.isnan(faithfulness) or np.isinf(faithfulness):
        #         faithfulness = 0
        #     faithfulnesses[i] = faithfulness
        # faithfulness_corr = np.mean(faithfulnesses)
        return faithfulness.item()


@time_function
def compute_faith_corr(
    model: Callable,
    rand_model: Callable,
    task: str,
    subset_size_faithfulness: int,
    x_test: torch.Tensor,
    attributions: torch.Tensor,
    rand_attrib: torch.Tensor,
    randmodel_attributions: torch.Tensor,
    label: Union[int, list, np.ndarray, torch.Tensor],
    metrics: dict,
    device: str = "cpu",
) -> dict:
    """Compute the faithfulness correlation metric.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task of the model
        subset_size_faithfulness (int): number of features to remove
        x_test (torch.Tensor): test data
        attributions (torch.Tensor): attributions for each instance
            for the base model
        rand_attrib (torch.Tensor): attributions for each instance
            for the random explainer
        randmodel_attributions (torch.Tensor): attributions for each instance
            for the random model
        label (int, list, np.ndarray, torch.Tensor): label(s) of interest
        metrics (dict): dictionary of metrics
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        dict: dict of metrics
    """
    n_features = x_test.shape[1]

    faith = FaithfulnessCorrelation(model, task, device)
    features_subset_size = min(subset_size_faithfulness, n_features)
    faith_score = faith.get_faithfulness(
        x_test, attributions, features_subset_size, label
    )
    metrics["FaithCorr"]["original"] = faith_score
    if rand_attrib is not None:
        rand_faith_score = faith.get_faithfulness(
            x_test, rand_attrib, features_subset_size, label
        )
        metrics["FaithCorr"]["random"] = rand_faith_score
    if rand_model is not None:
        randmodel_faith = FaithfulnessCorrelation(rand_model, task, device)
        randmodel_faith_score = randmodel_faith.get_faithfulness(
            x_test, randmodel_attributions, features_subset_size, label
        )
        metrics["FaithCorr"]["random model"] = randmodel_faith_score
    return metrics
