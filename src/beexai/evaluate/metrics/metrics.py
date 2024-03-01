from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from beexai.utils.convert import convert_to_tensor


class CustomMetric:
    """Base class for all metrics.

    Attributes:
        model (Callable): model to explain
        task (str): task to perform
        device (str): device to use

    Methods:
        select_output: select the output of the model for a given label
        get_predlb: get the prediction of the model for a given label
        choose_baseline: choose a baseline for removal based metrics
    """

    def __init__(self, model: Callable, task: str, device: str = "cpu"):
        assert task in [
            "classification",
            "regression",
        ], f"task must be in ['classification', 'regression'], found {task}"
        self.model = model
        self.task = task
        self.device = device

    def check_shape(self, x_in: torch.Tensor, attributions: torch.Tensor) -> None:
        """Check the shape of the attributions.

        Args:
            x_in (torch.Tensor): input data
            attributions (torch.Tensor): attributions
        """
        assert (
            x_in.ndim == 2 and attributions.ndim == 2
        ), f"""Input tensor and attributions tensor must be 2-dimensional.
            Found dimensions {x_in.ndim} and {attributions.ndim}"""
        assert (
            x_in.shape == attributions.shape
        ), f"""Input tensor and attributions tensor must have the same shape.
            Found shapes {x_in.shape} and {attributions.shape}"""

    def select_output(
        self,
        x_in: torch.Tensor,
        label: Optional[Union[int, list, np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the output of the model for a given label.
        If label is None, return the output of the model and
        the max argument of the probabilities (for classification).

        Args:
            x_in (torch.Tensor): input data
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.

        Returns:
            torch.Tensor: output of the model
            torch.Tensor: label of the output
        """
        if self.task == "classification":
            with torch.no_grad():
                pred = self.model.predict_proba(x_in)
            if isinstance(label, int):
                res = pred[:, label]
                max_arg = label * torch.ones(pred.shape[0], dtype=int)
            elif isinstance(label, (list, np.ndarray, torch.Tensor)):
                res = pred[torch.arange(pred.shape[0]), label]
                max_arg = label
            else:
                res = torch.max(pred, axis=1).values
                max_arg = torch.argmax(pred, dim=1)
        else:
            with torch.no_grad():
                res = self.model.predict(x_in).reshape(-1)
            max_arg = None
        res = convert_to_tensor(res, self.device)
        return res, max_arg

    def get_predlb(
        self,
        x_in: torch.Tensor,
        label: Optional[Union[int, list, np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get the prediction of the model for a given label.
        If label is None, return the prediction of the model for
        the max probability (for classification).

        Args:
            x_in (torch.Tensor): input data
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.

        Returns:
            torch.Tensor: prediction of the model
        """
        if isinstance(label, int):
            indexes = torch.ones(x_in.shape[0], dtype=int) * label
        elif isinstance(label, (list, np.ndarray, torch.Tensor)):
            indexes = label
        else:
            _, indexes = self.select_output(x_in)
        if self.task == "classification":
            with torch.no_grad():
                pred = self.model.predict_proba(x_in)
            pred = convert_to_tensor(pred, self.device)
            res = torch.zeros(pred.shape[0], device=self.device)
            for i in range(pred.shape[0]):
                res[i] = pred[i][indexes[i]]
        else:
            with torch.no_grad():
                res = self.model.predict(x_in).reshape(-1)
            res = convert_to_tensor(res, self.device)
        return res

    def choose_baseline(
        self,
        x_in: torch.Tensor,
        baseline: str,
        n_samples: int = 100,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Choose a baseline for removal based metrics.

        Args:
            x_in (torch.Tensor): input data
            baseline (str): baseline to use
            n_samples (int, optional): number of samples for
                multiple baselines. Defaults to 10.
            device (str, optional): device to use. Defaults to "cpu".

        Returns:
            torch.Tensor: baseline
        """
        authorized_baseline = [
            "mean",
            "median",
            "zero",
            "multiple",
            "normal",
            "uniform",
            "min",
            "max",
        ]
        assert (
            baseline in authorized_baseline
        ), f"baseline must be in {authorized_baseline}"
        res = torch.ones(x_in.shape, device=self.device)
        if baseline == "mean":
            res = torch.mul(torch.mean(x_in, axis=0), res)
        elif baseline == "median":
            res = torch.mul(torch.median(x_in, axis=0)[0], res)
        elif baseline == "zero":
            res = res * 0
        elif baseline == "multiple":
            samples = torch.randperm(x_in.shape[0])[:n_samples]
            res = torch.mean(x_in[samples], axis=0)
            res = torch.mul(res, torch.ones(x_in.shape, device=self.device))
        elif baseline == "uniform":
            for i in range(x_in.shape[1]):
                max_val = torch.max(x_in[:, i])
                min_val = torch.min(x_in[:, i])
                sample = torch.rand(x_in.shape[0])
                res[:, i] = sample * (max_val - min_val) + min_val
        elif baseline == "normal":
            for i in range(x_in.shape[1]):
                mean = torch.mean(x_in[:, i])
                std = torch.std(x_in[:, i])
                res[:, i] = torch.normal(mean, std, size=(x_in.shape[0],))
        elif baseline == "min":
            for i in range(x_in.shape[1]):
                res[:, i] = torch.min(x_in[:, i])
        elif baseline == "max":
            for i in range(x_in.shape[1]):
                res[:, i] = torch.max(x_in[:, i])
        res = res.to(device)
        return res
