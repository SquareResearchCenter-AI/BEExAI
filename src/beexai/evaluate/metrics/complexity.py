from typing import Callable

import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Complexity(CustomMetric):
    """Implementation of the complexity metric.

    Computes the complexity of the model by taking the entropy
    of the fractional contribution of each feature.

    References:
        - `Evaluating and Aggregating Feature-based Model Explanations
        <https://arxiv.org/abs/2005.00631>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        get_cmpl: computes the complexity of the model
    """

    def __total_contribution__(self, attribution: torch.Tensor) -> torch.Tensor:
        """Compute the total contribution of each instance."""
        return torch.sum(torch.abs(attribution), axis=1)

    def __fractional_contribution__(
        self, attribution: torch.Tensor, feature_i: int
    ) -> torch.Tensor:
        """Compute the fractional contribution of a given feature"""
        total_contrib = self.__total_contribution__(attribution)
        return torch.abs(attribution[:, feature_i]) / (total_contrib + 1e-8)

    def get_cmpl(self, attribution: torch.Tensor) -> torch.Tensor:
        """Computes the complexity of the model.

        Args:
            attribution (torch.Tensor): attributions for each instance

        Returns:
            torch.Tensor: array of complexity scores for each instance
        """
        self.check_shape(attribution, attribution)
        n_features = attribution.shape[1]
        complexity = torch.zeros(attribution.shape[0], device=self.device)
        for j in range(n_features):
            frac_contrib = self.__fractional_contribution__(attribution, j)
            # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
            # if frac_contrib[i] == 0:
            #   complexity[i] += 0
            # else:
            #   complexity[i] += -frac_contrib[i]*np.log(frac_contrib[i]+1e-8)
            complexity += -frac_contrib * torch.log(frac_contrib + 1e-8)
        complexity = complexity / n_features
        return torch.mean(complexity, axis=0)


@time_function
def compute_complex(
    model: Callable,
    rand_model: Callable,
    task: str,
    attributions: torch.Tensor,
    rand_attrib: torch.Tensor,
    randmodel_attributions: torch.Tensor,
    metrics: dict,
    device: str = "cpu",
) -> dict:
    """Computes the complexity of the base model, the random explainer
    and the random model.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task to perform
        attributions (torch.Tensor): feature attributions
        rand_attrib (torch.Tensor): random attributions
        randmodel_attributions (torch.Tensor): attributions of the
            random model
        metrics (dict): dictionary of metrics
        device (str, optional): device to use. Defaults to "cpu"

    Returns:
        dict: dict of metrics
    """
    comp = Complexity(model, task, device)
    comp_score = comp.get_cmpl(attributions)
    metrics["Complexity"]["original"] = comp_score.item()
    if rand_attrib is not None:
        rand_comp_score = comp.get_cmpl(rand_attrib)
        metrics["Complexity"]["random"] = rand_comp_score.item()
    if rand_model is not None:
        randmodel_comp = Complexity(rand_model, task, device)
        randmodel_comp_score = randmodel_comp.get_cmpl(randmodel_attributions)
        metrics["Complexity"]["random_model"] = randmodel_comp_score.item()
    return metrics
