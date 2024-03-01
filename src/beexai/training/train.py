"""Training models and evaluating their performance."""

from typing import Callable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     cross_validate)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from beexai.training.models import NeuralNetwork, NNModel
from beexai.utils.convert import convert_to_numpy
from beexai.utils.path import get_path


class Trainer:
    """Trainer class

    Attributes:
        models (dict): dictionary of available models
        model_name (str): name of the model
        model_params (dict): parameters of the model
        task (str): task to perform
        device (str): device to use
        model (callable): model object

    Methods:
        cross_val: cross validation for the model
        train: train the model
        get_metrics: get the metrics of the model
        save_model: save the model
        load_model: load the model

    Args:
        model_name (str): Name of the model from models dict. Must be one of
            'LogisticRegression', 'LinearRegression', 'DecisionTreeClassifier',
            'RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier',
            'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
            'XGBRegressor', 'NeuralNetwork', 'HistGradientBoostingClassifier',
            'HistGradientBoostingRegressor'
        task (str): "classification" or "regression".
        model_params (dict): Parameters for the model
        device (str): device to use. Defaults to "cpu".
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        model_params: Optional[dict] = None,
        device: str = "cpu",
    ):
        assert task in [
            "classification",
            "regression",
        ], f"Task must be either classification or regression, got {task}"
        self.models = {
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "XGBClassifier": XGBClassifier,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "XGBRegressor": XGBRegressor,
            "NeuralNetwork": NeuralNetwork,
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        }
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.task = task
        self.device = device
        assert (
            self.model_name in self.models
        ), f"Model name must be one of {self.models.keys()}"
        if model_name == "NeuralNetwork":
            self.model = NNModel(**self.model_params, device=device, task=task).to(
                device
            )
        else:
            self.model = self.models[model_name](**self.model_params)

    def cross_val(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        param_grid: Optional[dict] = None,
        scoring: Optional[str] = None,
        kfold: Union[int, KFold] = 5,
        search_type: str = "grid",
    ) -> Callable:
        """Cross validation for the model

        Args:
            x_train (pd.DataFrame): train set
            y_train (pd.DataFrame): target
            param_grid (dict, optional): grid search parameters. Defaults to None.
            scoring (str, optional): scoring metric. Defaults to None.
            kfold (Union[int, KFold], optional): number of folds or kfold object. Defaults to 5.
            search_type (str, optional): "grid" or "random". Defaults to "grid".

        Returns:
            callable: best model
        """
        assert search_type in ["grid", "random"]
        if self.model_name == "NeuralNetwork":
            x_train_copy = x_train.values
            y_train_copy = y_train.values
            if self.task == "classification":
                y_train_copy = y_train_copy.astype(np.int64)
            else:
                y_train_copy = y_train_copy.astype(np.float32)
        else:
            x_train_copy = x_train
            y_train_copy = y_train
        if not isinstance(param_grid, type(None)):
            if self.model_name == "NeuralNetwork":
                self.model.set_params(train_split=False, verbose=0)
            if search_type == "grid":
                grid_search = GridSearchCV(
                    self.model,
                    param_grid=param_grid,
                    cv=kfold,
                    scoring=scoring,
                    n_jobs=-1,
                )
            if search_type == "random":
                grid_search = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    cv=kfold,
                    scoring=scoring,
                    n_jobs=-1,
                    n_iter=10,
                )
            grid_search.fit(x_train_copy, y_train_copy)
            self.model = grid_search.best_estimator_
            scores = cross_validate(
                self.model,
                x_train_copy,
                y_train_copy,
                cv=kfold,
                scoring=scoring,
                return_estimator=True,
            )
            print(f"Best estimator: {grid_search.best_estimator_}")
            print(f"Best score: {scores['test_score'].mean()}")
            return self.model, grid_search.best_params_
        scores = cross_validate(
            self.model,
            x_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            return_estimator=True,
        )
        print(f"Best score: {scores['test_score'].mean()}")
        return self.model

    def train(
        self,
        x_train: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y_train: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        learning_rate: float = 0.005,
        epochs: int = 1000,
        loss_file: Optional[str] = None,
        x_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    ) -> Callable:
        """Perform training on the whole training set.

        Args:
            x_train (pd.DataFrame): x_train
            y_train (pd.DataFrame): y_train
            learning_rate (float, optional): learning rate. Defaults to 0.005.
            epochs (int, optional): number of epochs. Defaults to 1000.
            loss_file (str, optional): path to save the loss plot. Defaults to None.
            x_val (pd.DataFrame, optional): validation set. Defaults to None.
            y_val (pd.DataFrame, optional): validation target. Defaults to None.

        Returns:
            callable: trained model
        """
        if self.model_name == "NeuralNetwork":
            if not isinstance(x_train, np.ndarray):
                x_train_copy = x_train.values
            else:
                x_train_copy = x_train
            if not isinstance(y_train, np.ndarray):
                y_train_copy = y_train.values
            else:
                y_train_copy = y_train
            if self.task == "classification":
                y_train_copy = y_train_copy.astype(np.longlong)
            else:
                y_train_copy = y_train_copy.astype(np.float32)
            self.model = self.model.fit(
                x_train_copy,
                y_train_copy,
                learning_rate=learning_rate,
                epochs=epochs,
                loss_file=loss_file,
                x_val=x_val,
                y_val=y_val,
            )
        else:
            self.model.fit(x_train, y_train)
        return self.model

    def get_metrics(self, x: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Get metrics for the model. Accuracy and f1 score for
        classification, mse and r2 score for regression.

        Args:
            x (pd.DataFrame): test set
            y (pd.DataFrame): target

        Raises:
            Exception: Task must be either classification or regression

        Returns:
            dict: dictionary of metrics
        """
        if self.model_name == "NeuralNetwork" and not isinstance(x, np.ndarray):
            x_copy = x.values
            y_copy = y.values.squeeze()
        else:
            x_copy = x
            y_copy = y.squeeze()
        with torch.no_grad():
            pred = self.model.predict(x_copy)
        pred = convert_to_numpy(pred)
        if self.task == "classification":
            metrics = {
                "accuracy": accuracy_score(pred, y_copy),
                "f1 score": f1_score(pred, y_copy, average="weighted"),
            }
        if self.task == "regression":
            metrics = {
                "mse": mean_squared_error(pred, y_copy),
                "rmse": np.sqrt(mean_squared_error(pred, y_copy)),
                "mape": mean_absolute_percentage_error(pred, y_copy),
                "r2 score": r2_score(pred, y_copy),
            }
        return metrics

    def save_model(self, path: str):
        """Save the model"""
        path = get_path(path, check_dir=True)
        is_nn = self.model_name == "NeuralNetwork"
        if is_nn:
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load the model"""
        path = get_path(path)
        is_nn = self.model_name == "NeuralNetwork"
        if is_nn:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            self.model.to(self.device)
        else:
            self.model = joblib.load(path)


def test_all_models(
    task: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    params_dict: Optional[dict] = None,
) -> None:
    """Train and test all models on the whole training set

    Args:
        task (str): "classification" or "regression"
        x_train (pd.DataFrame): train set
        x_test (pd.DataFrame): test set
        y_train (pd.DataFrame): train target
        y_test (pd.DataFrame): test target
        params_dict (dict, optional): parameters for each model. Defaults to None.
    """
    params_dict = {} if params_dict is None else params_dict
    for model_name in params_dict.keys():
        print(f"Testing model: {model_name}")
        model = Trainer(model_name, task, params_dict[model_name])
        model.train(x_train, y_train)
        with torch.no_grad():
            pred = model.model.predict(x_test)
        if task == "classification":
            print(f"Accuracy: {accuracy_score(pred, y_test)}")
            print(f"F1 score: \n{f1_score(pred, y_test, average=None)}")
            print("\n")
        elif task == "regression":
            print(f"MSE: {mean_squared_error(pred, y_test)}")
            print(f"R2 Score: {r2_score(pred, y_test)}")
            print("\n")


def grid_search_all_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    task: str,
    params_dict: Optional[dict] = None,
    params_grid_dict: Optional[dict] = None,
    scoring: Optional[str] = None,
    kfold: Union[int, KFold] = 5,
    search_type: str = "grid",
) -> Tuple[dict, dict]:
    """Grid search for all models

    Args:
        x_train (pd.DataFrame): x_train
        y_train (pd.DataFrame): y_train
        task (str): "classification" or "regression"
        params_dict (dict, optional): parameters for each model. Defaults to None.
        params_grid_dict (dict, optional): grid search parameters for each model.
            Defaults to None.
        scoring (str, optional): scoring metric. Defaults to None.
        kfold (Union[int, KFold], optional): kfold object. Defaults to 5.
        search_type (str, optional): "grid" or "random". Defaults to "grid".

    Returns:
        Tuple[dict, dict]: best models and best parameters
    """
    best_models = {}
    best_params = {}
    params_dict = {} if params_dict is None else params_dict
    params_grid_dict = {} if params_grid_dict is None else params_grid_dict
    if params_dict == {}:
        params_dict = params_grid_dict.copy()
    for model_name in params_dict.keys():
        model = Trainer(model_name, task, params_dict[model_name])
        model, params = model.cross_val(
            x_train,
            y_train,
            param_grid=params_grid_dict[model_name],
            scoring=scoring,
            kfold=kfold,
            search_type=search_type,
        )
        best_models[model_name] = model
        best_params[model_name] = params
        print(f"Best params for {model_name}: {params}")
    return best_models, best_params
