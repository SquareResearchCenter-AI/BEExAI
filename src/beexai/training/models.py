"""Architectures for neural networks."""

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

from beexai.utils.convert import convert_to_tensor


class NeuralNetworkBlock(nn.Module):
    """Neural network block class."""

    def __init__(
        self,
        n_neurons: int = 32,
        batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(n_neurons, n_neurons)
        self.bn = nn.BatchNorm1d(n_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = batch_norm
        self.use_dropout = use_dropout

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear(x_in))
        if self.batch_norm:
            x = self.bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class NeuralNetwork(nn.Module):
    """Neural network class."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task: str,
        n_neurons: int = 32,
        batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.1,
        n_hidden_layers: int = 1,
    ):
        super().__init__()
        assert task in [
            "classification",
            "regression",
        ], f"task must be in ['classification', 'regression'], found {task}"
        if task == "regression":
            output_dim = 1
        self.linear1 = nn.Linear(input_dim, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.bn2 = nn.BatchNorm1d(n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.hidden_blocks = nn.Sequential(
            *[
                NeuralNetworkBlock(
                    n_neurons=n_neurons,
                    batch_norm=batch_norm,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_hidden_layers)
            ]
        )
        self.linear3 = nn.Linear(n_neurons, output_dim)
        self.task = task
        self.batch_norm = batch_norm
        self.use_dropout = use_dropout

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x_in))
        if self.batch_norm:
            x = self.bn1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        x = self.hidden_blocks(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.linear3(x)
        if self.task == "classification":
            return torch.softmax(x, dim=1)
        return x


class NNModel(NeuralNetwork):
    """Inherit from NeuralNetwork to overwrite fit and predict methods.

    Attributes:
        output_dim (int): output dimension
        device (str): device to use

    Methods:
        fit: fit the model
        predict: predict the output
        predict_proba: predict the output probabilities
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task: str,
        n_neurons: int = 32,
        device: str = "cpu",
        batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.1,
        n_hidden_layers: int = 1,
    ):
        super().__init__(
            input_dim,
            output_dim,
            task,
            n_neurons=n_neurons,
            batch_norm=batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            n_hidden_layers=n_hidden_layers,
        )
        self.output_dim = output_dim
        self.device = device

    def train_step(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        criterion: Any,
        optimizer: Any,
    ) -> float:
        """Train the model for one epoch.

        Args:
            x_train (torch.Tensor): features
            y_train (torch.Tensor): labels
            criterion (any): loss function
            optimizer (any): optimizer

        Returns:
            float: loss
        """
        y_pred = self.forward(x_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def val_step(
        self,
        x_val: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y_val: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        criterion: Any,
    ) -> float:
        """Validate the model for one epoch.

        Args:
            x_val (torch.Tensor): features
            y_val (torch.Tensor): labels
            criterion (any): loss function

        Returns:
            float: loss
        """
        x_val_copy = convert_to_tensor(x_val).float().to(self.device)
        y_val_copy = convert_to_tensor(y_val).to(self.device)
        if self.task == "classification":
            y_val_copy = y_val_copy.long().squeeze()
        with torch.no_grad():
            y_pred = self.forward(x_val_copy)
        loss = criterion(y_pred, y_val_copy)
        return loss.item()

    def fit(
        self,
        x_train: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y_train: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        learning_rate: float = 0.005,
        epochs: int = 1000,
        loss_file: Optional[str] = None,
        x_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    ) -> Any:
        x_train_copy = convert_to_tensor(x_train).float().to(self.device)
        y_train_copy = convert_to_tensor(y_train).to(self.device)
        if self.task == "classification":
            criterion = nn.CrossEntropyLoss()
            y_train_copy = y_train_copy.long().squeeze()
        elif self.task == "regression":
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        progress_bar = tqdm.tqdm(range(epochs))
        loss_history = []
        val_loss_history = []
        for _ in progress_bar:
            loss = self.train_step(x_train_copy, y_train_copy, criterion, optimizer)
            progress_bar.set_description(f"Loss: {loss:.3f}")
            loss_history.append(loss)
            if x_val is not None and y_val is not None:
                val_loss = self.val_step(x_val, y_val, criterion)
                val_loss_history.append(val_loss)
            _ = plt.figure(figsize=(12, 8))
            plt.plot(loss_history, label="train")
            if x_val is not None and y_val is not None:
                plt.plot(val_loss_history, label="val")
            plt.legend()
            if loss_file is not None:
                plt.savefig(loss_file)
            plt.close()
        return self

    def predict(
        self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x_test_copy = convert_to_tensor(x_test).float().to(self.device)
        if self.task == "classification":
            with torch.no_grad():
                res = self.forward(x_test_copy)
            res = torch.argmax(res, dim=1)
            return res
        if self.task == "regression":
            with torch.no_grad():
                res = self.forward(x_test_copy)
            return res
        return torch.Tensor([])

    def predict_proba(
        self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        assert (
            self.task == "classification"
        ), f"""predict_proba is only available for classification,
            found {self.task}"""
        x_test_copy = convert_to_tensor(x_test).float().to(self.device)
        with torch.no_grad():
            res = self.forward(x_test_copy)
        return res
