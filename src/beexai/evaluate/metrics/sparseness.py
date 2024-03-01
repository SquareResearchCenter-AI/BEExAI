from typing import Callable

import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Sparseness(CustomMetric):
    """Implementation of the sparseness metric.

    Computes the sparseness of the model based on the Gini index.

    References:
        - `Synthetic Benchmarks for Scientific Research in
        Explainable Machine Learning <https://arxiv.org/abs/2106.12543>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_sparsity: computes the sparseness of the model
    """

    def get_sparsity(self, attribution: torch.Tensor) -> torch.Tensor:
        """Computes the sparseness of the model.

        Args:
            attribution (torch.Tensor): attributions for each instance

        Returns:
            torch.Tensor: array of sparseness scores for each instance
        """
        self.check_shape(attribution, attribution)
        n_features = attribution.shape[1]
        spars = torch.zeros(attribution.shape[0], device=self.device)
        attrib_sum = torch.sum(torch.abs(attribution), axis=1)
        for j in range(1, n_features + 1):
            spars += (n_features - j + 0.5) * torch.abs(attribution[:, j - 1])
        spars = 1 - 2 * spars / (attrib_sum * n_features + 1e-8)
        return torch.mean(spars, axis=0)


@time_function
def compute_spar(
    model: Callable,
    rand_model: Callable,
    task: str,
    attributions: torch.Tensor,
    rand_attrib: torch.Tensor,
    rand_model_attributions: torch.Tensor,
    metrics: dict,
    device="cpu",
) -> dict:
    """Compute the sparseness metric.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task of the model
        attributions (torch.Tensor): attributions for base model and
            base explainer
        rand_attrib (torch.Tensor): random attributions
        rand_model_attributions (torch.Tensor): attributions for reference
            model and base explainer
        metrics (dict): dictionary of metrics
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        dict: dict of metrics
    """
    spars = Sparseness(model, task, device)
    spars_score = spars.get_sparsity(torch.sort(torch.abs(attributions), axis=1)[0])
    metrics["Sparseness"]["original"] = spars_score.item()
    if rand_attrib is not None:
        rand_spars_score = spars.get_sparsity(
            torch.sort(torch.abs(rand_attrib), axis=1)[0]
        )
        metrics["Sparseness"]["random"] = rand_spars_score.item()
    if rand_model is not None:
        randmodel_spars = Sparseness(rand_model, task, device)
        randmodel_spars_score = randmodel_spars.get_sparsity(
            torch.sort(torch.abs(rand_model_attributions), axis=1)[0]
        )
        metrics["Sparseness"]["random_model"] = randmodel_spars_score.item()
    return metrics
