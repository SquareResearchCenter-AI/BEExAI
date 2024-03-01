"""Creation of dataset splits and encoding/scaling of features"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)


class Scaler:
    """
    Class for scaling the data.

    Attributes:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name
        categorical_cols (list): list of categorical columns
        x_num_scaler_name (str): scaler to use for x. Must be either
            None or one of standard, minmax, quantile_normal, quantile_uniform,
            maxabs, robust. Defaults to None.
        x_cat_encoder_name (str): scaler to use for x. Must be either None or
            one of labelencoder or onehotencoder. Defaults to None.
        y_scaler_name (str): scaler to use for y. Must be either None or one
            of standard, minmax, quantile_normal, quantile_uniform, maxabs,
            robust or labelencoder. Defaults to None.
        cat_not_to_onehot (List[str]): list of categorical columns not to one
            hot encode. Defaults to [].
        scalers (dict): dictionary of possible scalers

    Methods:
       encode_categorical: encode categorical columns in one hot or label encoding
       do_scaling: process data from categorical features first to numerical features scaling

    Args:
        df (pd.DataFrame): input dataframe
        target_col (str, optional): target column name. Defaults to None.
        x_num_scaler_name (Optional[str], optional): scaler to use for x.
            Must be either None or one of standard, minmax, quantile_normal,
            quantile_uniform, maxabs, robust. Defaults to None.
        x_cat_encoder_name (Optional[str], optional): scaler to use for x.
            Must be either None or one of labelencoder or onehotencoder. Defaults to None.
        y_scaler_name (Optional[str], optional): scaler to use for y.
            Must be either None or one of standard, minmax, quantile_normal,
            quantile_uniform, maxabs, robust or labelencoder. Defaults to None.
        cat_not_to_onehot (Optional[List[str]], optional): list of
            categorical columns not to one hot encode. Defaults to [].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        x_num_scaler_name: Optional[str] = None,
        x_cat_encoder_name: Optional[str] = None,
        y_scaler_name: Optional[str] = None,
        cat_not_to_onehot: Optional[List[str]] = [],
    ):
        self.df = df
        self.target_col = target_col
        self.categorical_cols = list(set(df.select_dtypes(include=["object"]).columns))
        self.x_num_scaler_name = x_num_scaler_name
        self.x_cat_encoder_name = x_cat_encoder_name
        self.y_scaler_name = y_scaler_name
        self.cat_not_to_onehot = cat_not_to_onehot
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "quantile_normal": QuantileTransformer(output_distribution="normal"),
            "quantile_uniform": QuantileTransformer(output_distribution="uniform"),
            "maxabs": MaxAbsScaler(),
            "robust": RobustScaler(),
            "labelencoder": LabelEncoder(),
            "ordinalencoder": OrdinalEncoder(),
            "onehot": OneHotEncoder(),
        }
        assert (
            self.x_num_scaler_name is None or self.x_num_scaler_name in self.scalers
        ), "x_num_scaler_name must be None or \
         one of standard, minmax, quantile_normal, quantile_uniform, maxabs, robust or ordinalencoder"
        assert self.x_cat_encoder_name is None or self.x_cat_encoder_name in [
            "labelencoder",
            "onehot",
            "ordinalencoder",
        ], "x_cat_encoder_name must be None or one of labelencoder or onehotencoder or ordinalencoder"
        assert (
            self.y_scaler_name is None or self.y_scaler_name in self.scalers
        ), "y_scaler_name must be None or \
         one of standard, minmax, quantile_normal, quantile_uniform, maxabs, robust or labelencoder"

    def __cat_cols__(self, x: pd.DataFrame) -> List[str]:
        """Check if an array represents categorical data"""
        cat_cols_names = []
        for col in x.columns:
            if (
                x[col].dtype in [np.int64, np.int32, np.int16, np.int8]
                and np.max(np.abs(x[col])) < 2
            ):
                cat_cols_names.append(col)
        return cat_cols_names

    def encode_categorical(
        self,
        x_train: pd.DataFrame,
        x_test: Optional[pd.DataFrame],
        x_cat_encoder: Optional[str] = "labelencoder",
        cat_not_to_onehot: Optional[List[str]] = [],
    ) -> pd.DataFrame:
        """Encode categorical columns with LabelEncoder or OneHotEncoder

        Args:
            x_train (pd.DataFrame): train dataframe with categorical columns
            x_test (pd.DataFrame): test dataframe with categorical columns
            x_cat_encoder (Optional[str], optional): encoding type for categorical columns
            cat_not_to_onehot (Optional[List[str]], optional): list of
                categorical columns not to one hot encode. For example if the
                dimensionality is too high. Defaults to [].

        Returns:
            pd.DataFrame: dataframe with encoded categorical columns
        """
        categorical_cols = list(set(self.categorical_cols) - set([self.target_col]))
        x_traincp, x_testcp = x_train.copy(), x_test.copy()
        for col in categorical_cols:
            if col not in cat_not_to_onehot and x_cat_encoder == "onehot":
                one = OneHotEncoder()
                one_hot_train = one.fit_transform(
                    x_traincp[col].values.reshape(-1, 1)
                ).toarray()
                x_traincp = x_traincp.drop(col, axis=1)
                for i in range(one_hot_train.shape[1]):
                    x_traincp[col + "_" + str(i)] = one_hot_train[:, i].astype(np.int32)
                one_hot_test = one.transform(
                    x_testcp[col].values.reshape(-1, 1)
                ).toarray()
                x_testcp = x_testcp.drop(col, axis=1)
                for i in range(one_hot_test.shape[1]):
                    x_testcp[col + "_" + str(i)] = one_hot_test[:, i].astype(np.int32)
            else:
                le = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                x_traincp[col] = le.fit_transform(
                    x_traincp[col].values.reshape(-1, 1)
                ).astype(np.int32)
                x_testcp[col] = le.transform(
                    x_testcp[col].values.reshape(-1, 1)
                ).astype(np.int32)
        return x_traincp, x_testcp

    def do_scaling(
        self,
        x_train: pd.DataFrame,
        x_test: Optional[pd.DataFrame],
        y_train: pd.Series,
        y_test: Optional[pd.Series],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits.

        Args:
            x_train (pd.DataFrame): train dataframe
            x_test (pd.DataFrame): test dataframe
            y_train (pd.Series): train targets
            y_test (pd.Series): test targets

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
        x_num_scaler = (
            self.scalers[self.x_num_scaler_name]
            if self.x_num_scaler_name is not None
            else None
        )
        y_scaler = (
            self.scalers[self.y_scaler_name] if self.y_scaler_name is not None else None
        )

        if self.x_cat_encoder_name is not None:
            x_traincp, x_testcp = self.encode_categorical(
                x_train, x_test, self.x_cat_encoder_name, self.cat_not_to_onehot
            )
        else:
            x_traincp, x_testcp = x_train, x_test
        y_traincp, y_testcp = y_train, y_test

        self.categorical_cols = self.__cat_cols__(x_traincp)
        x_num_cols = x_traincp.columns[~x_traincp.columns.isin(self.categorical_cols)]
        x_train_num = x_traincp.loc[:, ~x_traincp.columns.isin(self.categorical_cols)]
        x_train_num = x_train_num.astype(np.float32)
        x_test_num = x_testcp.loc[:, ~x_testcp.columns.isin(self.categorical_cols)]
        x_test_num = x_test_num.astype(np.float32)
        if self.x_num_scaler_name is not None and x_train_num.shape[1] > 0:
            x_train_num = x_num_scaler.fit_transform(x_train_num)
            x_train_num = pd.DataFrame(x_train_num, columns=x_num_cols)
            x_test_num = x_num_scaler.transform(x_test_num)
            x_test_num = pd.DataFrame(x_test_num, columns=x_num_cols)
        if self.y_scaler_name is not None:
            if self.y_scaler_name == "labelencoder" and y_traincp.dtype in [
                np.int64,
                np.int32,
                str,
            ]:
                y_traincp = y_scaler.fit_transform(y_traincp)
                y_traincp = pd.DataFrame(y_traincp, columns=[self.target_col])
                y_traincp = y_traincp - y_traincp.min()
                y_testcp = y_scaler.transform(y_testcp)
                y_testcp = pd.DataFrame(y_testcp, columns=[self.target_col])
                y_testcp = y_testcp - y_testcp.min()
            else:
                y_traincp = y_scaler.fit_transform(y_traincp.values.reshape(-1, 1))
                y_traincp = pd.DataFrame(y_traincp.flatten(), columns=[self.target_col])
                y_testcp = y_scaler.transform(y_testcp.values.reshape(-1, 1))
                y_testcp = pd.DataFrame(y_testcp.flatten(), columns=[self.target_col])
        x_train_cat = x_traincp.loc[:, x_traincp.columns.isin(self.categorical_cols)]
        x_train_cat = x_train_cat.reset_index(drop=True)
        x_train_num = x_train_num.reset_index(drop=True)
        x_traincp = x_train_num.join(x_train_cat)
        y_traincp = y_traincp.reset_index(drop=True)
        x_test_cat = x_testcp.loc[:, x_testcp.columns.isin(self.categorical_cols)]
        x_test_cat = x_test_cat.reset_index(drop=True)
        x_test_num = x_test_num.reset_index(drop=True)
        x_testcp = x_test_num.join(x_test_cat)
        y_testcp = y_testcp.reset_index(drop=True)
        return x_traincp, x_testcp, y_traincp, y_testcp


class Dataset:
    """Dataset class to create train/test splits.

    Attributes:
        target_name (str): name of the target column
        data (pd.DataFrame): input data
        target (pd.Series): target column

    Methods:
        get_train_test: create train/test splits
        get_classes_num: get the number of classes for the task

    Args:
        data (pd.DataFrame): input data
        target_name (str): name of the target column
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_name: str,
    ):
        self.target_name = target_name
        self.data = data.drop(target_name, axis=1)
        self.target = data[target_name]

    def get_train_test(
        self, test_size: float = 0.2, scaler_params: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits.

        Args:
            test_size (float, optional): test size. Defaults to 0.2.
            scaler_params (Optional[Dict[str, str]], optional): scaling parameters.
                Defaults to None.

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        x_train, x_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=test_size
        )
        if scaler_params is not None:
            scaler = Scaler(x_train, self.target_name, **scaler_params)
            x_train, x_test, y_train, y_test = scaler.do_scaling(
                x_train, x_test, y_train, y_test
            )
        x_train, x_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
        return x_train, x_test, y_train, y_test

    def get_train_val(
        self, x_train: pd.DataFrame, y_train: pd.Series, val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/val splits.

        Args:
            x_train (pd.DataFrame): input data
            y_train (pd.Series): target column
            val_size (float, optional): validation size. Defaults to 0.2.

        Returns:
            tuple: x_train, x_val, y_train, y_val
        """
        assert 0 < val_size < 1, "val_size must be between 0 and 1"
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size
        )
        x_train, x_val = x_train.reset_index(drop=True), x_val.reset_index(drop=True)
        y_train, y_val = y_train.reset_index(drop=True), y_val.reset_index(drop=True)
        return x_train, x_val, y_train, y_val

    def get_classes_num(self, task: str) -> int:
        """Get the number of classes for the task. Return 1 for regression.

        Args:
            task (str): task type

        Returns:
            int: number of classes
        """
        assert task in [
            "classification",
            "regression",
        ], "task must be in ['classification', 'regression']"
        if task == "regression":
            return 1
        return np.unique(self.target).shape[0]
