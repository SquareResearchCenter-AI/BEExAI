from typing import Callable, Optional, Union

import numpy
import torch

from beexai.evaluate.metrics.metrics import CustomMetric
from beexai.explanation.explaining import GeneralExplainer
from beexai.utils.time_seed import time_function


class Sensitivity(CustomMetric):
    """Implementation of the sensitivity metric.

    Computes the sensitivity of the model by adding significant
    noise to the input and compute the difference in attributions
    between the original input and the input with a small perturbation.

    References:
        - `On the (In)fidelity and Sensitivity for Explanations
        <https://arxiv.org/abs/1901.09392>`

    Attributes:
        model (callable): model to explain
        task (str): task to perform
        device (str): device to use
        explainer (object): explainer to use
        radius (float): radius of the uniform distribution to generate
            the noise

    Methods:
        get_sens: computes the sensitivity of the model
    """

    def __init__(
        self,
        model: Callable,
        task: str,
        device: str,
        explainer: GeneralExplainer,
        radius=0.5,
    ):
        super().__init__(model, task, device)
        self.explainer = explainer
        self.radius = radius

    def __get_noises__(self, x_in: torch.Tensor, k: int = 5):
        """Generate k noises from a uniform distribution with
        mean 0 and radius self.radius."""
        n_shape = (k, x_in.shape[0], x_in.shape[1])
        noises = torch.rand(n_shape, device=self.device)
        radius = torch.tensor(self.radius, device=self.device)
        if radius.ndim == 0:
            radius = radius.repeat(x_in.shape[1])
        noises = 2 * radius[None, None, :] * noises - radius[None, None, :]
        return noises

    def get_sens(
        self,
        x_in: torch.Tensor,
        label: Optional[Union[int, list, torch.Tensor]] = None,
        attributions: Optional[torch.Tensor] = None,
    ) -> float:
        """Computes the sensitivity of the model.

        Args:
            x_in (torch.Tensor): input to compute the sensitivity score
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            attributions (torch.Tensor, optional): attributions for each
                instance. Defaults to None. If None, the attributions are computed
                using the explainer.

        Returns:
            float: sensitivity score
        """
        self.check_shape(x_in, x_in)
        noises = self.__get_noises__(x_in)
        _, target = self.select_output(x_in, label=label)
        if isinstance(target, (numpy.ndarray, list)):
            target = torch.tensor(target, device=self.device)
        if attributions is None:
            attributions = self.explainer.explain(x_in, label=target)
        self.check_shape(x_in, attributions)
        sensitivities = torch.zeros((len(noises), x_in.shape[0]), device=self.device)
        for _, j in enumerate(range(len(noises))):
            noise = noises[j]
            pert_in = x_in - noise
            pert_att = self.explainer.explain(pert_in, label=target)
            sensitivity = torch.norm(attributions - pert_att, dim=1)
            rho = torch.norm(noise.flatten())
            sensitivities[j] = sensitivity / rho
        all_sens = torch.max(sensitivities, dim=0).values
        return 100 * torch.mean(all_sens, axis=0).item()


@time_function
def compute_sens(
    model: Callable,
    rand_model: Callable,
    task: str,
    x_test: torch.Tensor,
    label: Union[int, list, torch.Tensor],
    metrics: dict,
    exp: GeneralExplainer,
    randmodel_exp: GeneralExplainer,
    device: str = "cpu",
    use_rand: bool = True,
    attributions=None,
    rand_attributions=None,
    randmodel_attributions=None,
    radius=0.5,
) -> dict:
    """Computes the sensitivity score of the model.

    Args:
        model (callable): base model
        rand_model (callable): reference model (random model)
        task (str): task to perform
        x_test (torch.Tensor): test data
        label (int, list, np.ndarray, torch.Tensor): label(s) of interest
        metrics (dict): dictionary of metrics
        exp (GeneralExplainer): base explainer
        randmodel_exp (GeneralExplainer): explainer for the random model
        device (str, optional): device to use. Defaults to "cpu".
        use_rand (bool, optional): whether to use the random explainer.
            Defaults to True.
        attributions (torch.Tensor, optional): attributions for each
            instance. Defaults to None.
        rand_attributions (torch.Tensor, optional): attributions for each
            instance for the random explainer. Defaults to None.
        randmodel_attributions (torch.Tensor, optional): attributions for each
            instance for the random model. Defaults to None.
        radius (float, optional): radius of the uniform distribution to generate
            the noise. Defaults to 0.5.

    Returns:
        dict: dict of metrics
    """
    sens = Sensitivity(model, task, device, exp, radius=radius)
    sens_score = sens.get_sens(x_test, label, attributions)
    metrics["Sensitivity"]["original"] = sens_score

    if use_rand:
        rand_sens_score = sens.get_sens(x_test, label, rand_attributions)
        metrics["Sensitivity"]["random"] = rand_sens_score

    if rand_model is not None:
        randmodel_sens = Sensitivity(
            rand_model, task, device, randmodel_exp, radius=radius
        )
        randmodel_sens_score = randmodel_sens.get_sens(
            x_test, label, randmodel_attributions
        )
        metrics["Sensitivity"]["random_model"] = randmodel_sens_score
    return metrics
