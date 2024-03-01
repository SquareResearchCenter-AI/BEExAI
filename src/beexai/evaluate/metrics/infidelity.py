from typing import Callable, Optional, Union

import numpy as np
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.utils.time_seed import time_function


class Infidelity(CustomMetric):
    """Implementation of the infidelity metric.

    Computes the infidelity of the model by adding significant
    noise to the input and compute the mean-squared error between
    the pertubation applicated to the attribution and
    the difference in prediction original input and perturbed input.

    References:
        - `On the (In)fidelity and Sensitivity for Explanations
        <https://arxiv.org/abs/1901.09392>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use
        std (float): std of the noise

    Methods:
        get_inf: computes the infidelity of the model
    """

    def __init__(
        self, model: Callable, task: str, std: float = 0.003, device: str = "cpu"
    ):
        super().__init__(model, task, device)
        self.std = std

    def __get_noises__(self, x_in: torch.Tensor, k: int = 5):
        """Generate k noises from a normal distribution with
        mean 0 and std self.std."""
        n_shape = (k, x_in.shape[0], x_in.shape[1])
        stds = torch.tensor(self.std, device=self.device).float()
        if stds.ndim == 0:
            stds = [stds]
        all_noises = torch.concatenate(
            [torch.normal(0, std, n_shape, device=self.device) for std in stds]
        )
        return x_in - all_noises

    def get_inf(
        self,
        x_in: torch.Tensor,
        attributions: torch.Tensor,
        label: Optional[Union[int, list, torch.Tensor, np.ndarray]] = None,
    ) -> float:
        """Computes the infidelity of the model.

        Args:
            x_in (torch.Tensor): input data
            attributions (torch.Tensor): attributions for each instance
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.

        Returns:
            float: infidelity score
        """
        self.check_shape(x_in, attributions)
        pred, max_arg = self.select_output(x_in, label=label)
        pert = self.__get_noises__(x_in)
        infs = torch.zeros((x_in.shape[0], len(pert)), device=self.device)
        for j, noise in enumerate(pert):
            pert_in = x_in - noise
            pert_pred = self.get_predlb(pert_in, max_arg)
            diff = (pred - pert_pred).squeeze()
            if self.task == "regression":
                diff = torch.abs(diff)
            perturbed_term = torch.einsum("ij, ji->i", noise, attributions.T)
            inf = (perturbed_term - diff) ** 2
            infs[:, j] = inf
        # **PREVIOUS IMPLEMENTATION LESS EFFICIENT BUT MORE READABLE**
        # for i in range(X.shape[0]):
        #     infidelities = np.zeros(len(pert))
        #     for j,perturbation in enumerate(pert):
        #         repeated_perturbation = perturbation[i]
        #         perturbed_input = (X[i] - repeated_perturbation).unsqueeze(0)
        #         perturbed_pred = self.get_predlb(perturbed_input,max_arg)
        #         diff = pred[i] - perturbed_pred
        #         attribute = attribution_tensor[i]
        #         perturbed_term = torch.matmul(
        #             perturbation[i],attribute).detach().cpu().numpy()
        #         infidelity = (perturbed_term - diff)**2
        #         infidelities[j] = infidelity.item()
        #     infidelity = np.mean(infidelities,axis=0)
        #     mean_inf[i] = infidelity
        mean_inf = torch.mean(infs, axis=1)
        return torch.mean(mean_inf, axis=0).item()


@time_function
def compute_inf(
    model: Callable,
    rand_model: Callable,
    task: str,
    x_test: torch.Tensor,
    attributions: torch.Tensor,
    rand_attrib: torch.Tensor,
    randmodel_attributions: torch.Tensor,
    label: Union[int, list, torch.Tensor, np.ndarray],
    metrics: dict,
    device: str = "cpu",
    inf_std: float = 0.003,
) -> dict:
    """Compute the infidelity metric.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task of the model
        x_test (torch.Tensor): test data
        attributions (torch.Tensor): attributions for base model
        rand_attrib (torch.Tensor): random attributions
        randmodel_attributions (torch.Tensor): attributions for reference model
        label (Union[int, list, np.ndarray, torch.Tensor]): label(s) of interest
        metrics (dict): dictionary of metrics
        device (str, optional): device to use. Defaults to "cpu"
        inf_std (float, optional): std of the noise.
            Defaults to 0.003.

    Returns:
        dict: dict of metrics
    """
    inf = Infidelity(model, task, inf_std, device)
    inf_score = inf.get_inf(x_test, attributions, label)
    metrics["Infidelity"]["original"] = inf_score

    if rand_attrib is not None:
        rand_inf_score = inf.get_inf(x_test, rand_attrib, label)
        metrics["Infidelity"]["random"] = rand_inf_score

    if rand_model is not None:
        randmodel_inf = Infidelity(rand_model, task, inf_std, device)
        randmodel_inf_score = randmodel_inf.get_inf(
            x_test, randmodel_attributions, label
        )
        metrics["Infidelity"]["random_model"] = randmodel_inf_score
    return metrics
