"""General explainer classes and subclassed explainers"""

import abc
import os
from typing import Callable, Optional, Union

import captum
import joblib
import numpy as np
import pandas as pd
import torch
from captum.attr import (DeepLift, FeatureAblation, InputXGradient,
                         IntegratedGradients, KernelShap, Lime, Saliency,
                         ShapleyValueSampling)


class GeneralExplainer:
    """General explainer class.

    Attributes:
        model (Callable): model to explain
        task (str): task to perform

    Methods:
        init_explainer: initialize explainer
        explain: explain a single instance
        feature_order: get indexes of features sorted by importance

    Args:
        model (Callable): model to explain
        task (str): task to perform
    """

    def __init__(self, model: Callable, task: str):
        assert task in [
            "classification",
            "regression",
        ], f"task must be in ['classification', 'regression'], found {task}"
        self.model = model
        self.task = task

    @abc.abstractmethod
    def init_explainer(self, *args, **kwargs) -> None:
        """Initialize explainer."""

    @abc.abstractmethod
    def explain(
        self, x_test: Union[pd.DataFrame, torch.Tensor, np.ndarray], *args, **kwargs
    ) -> torch.Tensor:
        """Explain a single instance.

        Args:
            x_test (pd.DataFrame): test set
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            torch.Tensor: array of attributions (n_samples, n_features)
        """

    def feature_order(self, attributions: torch.Tensor) -> torch.Tensor:
        """Get indexes of features sorted by importance.

        Args:
            attributions (torch.Tensor): array of attributions (n_samples, n_features)

        Returns:
            torch.Tensor: array of indexes of features sorted by importance
        """
        if isinstance(attributions, np.ndarray):
            return np.argsort(-(attributions), axis=1)
        if isinstance(attributions, torch.Tensor):
            return torch.argsort(-(attributions), axis=1)
        raise ValueError("Attributions must be a torch.Tensor or a np.ndarray")

    def compute_attributions(
        self,
        x_in: pd.DataFrame,
        data_name: str,
        model_name: str,
        method_name: str,
        folder_path: str,
        preds: Optional[np.ndarray] = None,
        save: bool = False,
        use_abs: bool = False,
    ) -> torch.Tensor:
        """Save the attributions of a model in folder "folder_path/attributions/data_name"
        and for each label in "folder_path/attributions/data_name/model_name_method_name_label.pkl".
        If attributions are already saved, they are loaded from the same folder.

        Args:
            x_in (pd.DataFrame): input data
            data_name (str): name of the dataset
            model_name (str): name of the model
            method_name (str): name of the method
            folder_path (str): path of the folder where to save the attributions
                or retrieve existing attributions if previously saved.
            preds (np.ndarray, optional): predictions of the model. Defaults to None.
            save (bool, optional): whether to save the attributions. Defaults to False.
            use_abs (bool, optional): whether to use the absolute value of the attributions.
                Defaults to False.

        Returns:
            torch.Tensor: tensor of attributions (n_samples, n_features)
        """
        assert hasattr(self, "explainer"), "Explainer not initialized"
        assert hasattr(self, "explain"), "Explainer must have an explain method"
        att_folder = f"{folder_path}/attributions/{data_name}/"
        suffix = f"{model_name}_{method_name}.pkl"
        if os.path.exists(att_folder + suffix):
            attribution = joblib.load(att_folder + suffix)
        else:
            attribution = self.explain(x_in, label=preds, absolute=use_abs)
            if save:
                if not os.path.exists(att_folder):
                    os.makedirs(att_folder)
                joblib.dump(attribution, att_folder + suffix)
        return attribution


class CaptumExplainer(GeneralExplainer):
    """Captum explainer class.

    Attributes:
        model (Callable): model to explain
        task (str): task to perform
        method (str): method to use
        sklearn (bool): whether to use a sklearn model
        explainer (captum.attr.Attribution): explainer
        all_methods (dict): all methods available
        device (str): device to use

    Methods:
        init_explainer: initialize explainer
        explain: explain a single instance

    Args:
        model (Callable): model to explain
        task (str): task to perform
        method (str): method to use. Must be one of the following:
            DeepLift, IntegratedGradients, Saliency, ShapleyValueSampling,
            KernelShap, InputXGradient, FeatureAblation, Lime
        sklearn (bool, optional): whether to use a sklearn model.
            Defaults to False.
        device (str, optional): device to use. Defaults to "cpu".
    """

    def __init__(
        self,
        model: Callable,
        task: str,
        method: str,
        sklearn: bool = False,
        device: str = "cpu",
    ):
        super().__init__(model, task)
        self.method = method
        self.sklearn = sklearn
        self.explainer = None
        self.all_methods = {
            "DeepLift": DeepLift,
            "IntegratedGradients": IntegratedGradients,
            "Saliency": Saliency,
            "ShapleyValueSampling": ShapleyValueSampling,
            "KernelShap": KernelShap,
            "InputXGradient": InputXGradient,
            "FeatureAblation": FeatureAblation,
            "Lime": Lime,
        }
        assert method in self.all_methods, (
            f"Method {method} not available. Choose one of the following: "
            f"{list(self.all_methods)}"
        )
        assert sklearn is False or method not in [
            "DeepLift",
            "IntegratedGradients",
            "Saliency",
            "InputXGradient",
        ], f"""Method {method} not available for sklearn models.
            Choose one of the following: ['ShapleyValueSampling', 'KernelShap',
            'FeatureAblation', 'Lime']"""
        self.device = device

    def __forward_wrapper__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Wrapper for sklearn model to convert the input in tensors."""
        x_in = tensor.detach().cpu().numpy()
        if self.task == "regression":
            with torch.no_grad():
                output = self.model.predict(x_in)
        elif self.task == "classification":
            with torch.no_grad():
                output = self.model.predict_proba(x_in)
        return torch.from_numpy(output)

    def init_explainer(self) -> captum.attr.Attribution:
        """Initialize Captum explainer.

        Returns:
            captum.attr.Attribution: explainer
        """
        if self.sklearn:
            explainer = self.all_methods[self.method](self.__forward_wrapper__)
        else:
            explainer = self.all_methods[self.method](self.model)
        self.explainer = explainer
        return explainer

    def explain(
        self,
        x_test: pd.DataFrame,
        label: Optional[Union[int, list, torch.Tensor, np.ndarray]] = None,
        absolute: bool = False,
    ) -> torch.Tensor:
        """Explain the whole set.

        Args:
            x_test (pd.DataFrame): test set
            label (int, list, np.ndarray, torch.Tensor, optional): label(s) of interest.
                Defaults to None. A list of labels for each instance can be provided.
            absolute (bool, optional): whether to use the absolute value of the attributions.
                Defaults to False.

        Returns:
            torch.Tensor: array of attributions (n_samples, n_features)
        """
        assert self.explainer is not None, "Explainer not initialized"
        x_tensor_test = x_test
        if isinstance(x_test, pd.DataFrame):
            x_tensor_test = torch.tensor(x_tensor_test.values).float()
        elif isinstance(x_test, np.ndarray):
            x_tensor_test = torch.tensor(x_tensor_test).float()
        if self.task == "regression":
            target = None
        elif self.task == "classification" and label is None:
            target = self.model.predict(x_tensor_test)
        else:
            target = label
        if self.task == "classification" and not isinstance(target, int):
            if isinstance(target, (np.ndarray, list)):
                target = torch.tensor(target, device=self.device)
            target = target.long()
        x_tensor_test = x_tensor_test.to(self.device)
        if self.method in ["Lime", "ShapleyValueSampling", "KernelShap"]:
            attributions = self.explainer.attribute(
                x_tensor_test, target=target, perturbations_per_eval=32
            )
        else:
            attributions = self.explainer.attribute(x_tensor_test, target=target)
        attributions = attributions.float()
        if absolute:
            attributions = torch.abs(attributions)
        attributions = attributions / torch.norm(attributions)
        return attributions
