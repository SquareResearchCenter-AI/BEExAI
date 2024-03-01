"""Provides a fast way to load data and preprocess it"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from yaml.loader import SafeLoader

from beexai.utils.path import get_path


class LoadData:
    """Load data from a csv file and return a dataframe

    Attributes:
        path (str): path to the csv file

    Methods:
        load_csv: load the csv file

    Args:
        path (str): path to the csv file
    """

    def __init__(self, path: str):
        self.path = get_path(path)

    def load_csv(self, keep_index: bool = False) -> pd.DataFrame:
        """Load the csv file

        Args:
            keep_index (bool): whether to keep the index or not. Defaults to False.

        Returns:
            pd.DataFrame: dataframe
        """
        if keep_index:
            return pd.read_csv(self.path, index_col=0)
        return pd.read_csv(self.path)


class Preprocessor:
    """Preprocess the data by deleting entries, adding new columns,
    converting to datetime and adding date infos.

    Attributes:
        df (pd.DataFrame): input dataframe
        encoder (object): encoder to use
        target_col (str): target column name
        datetime_cols (list): list of datetime columns
        values_to_delete (list): list of tuples (col_name,value to delete)
        add_cols (list): list of tuples (new_col_name,new_col_value,
            cast_to_type)
        cols_to_delete (list): list of columns to delete

    Methods:
        delete_entries: delete entries from the dataframe
        add_entries: add new colums to the dataframe with new values.
            These values are combinations of existing columns.
        convert_to_datetime: convert columns to datetime
        add_date_infos: add year, month, day and hour to the dataframe
        preprocess: preprocess the data
        save_cleaned_data: save the cleaned data

    Args:
        df (pd.DataFrame): input dataframe
        target_col (str, optional): target column name. Defaults to None.
        values_to_delete (List[Tuple[str,str]], optional):
            (col_name,value to delete) values to delete from the dataframe.
            Defaults to None.
        datetime_cols (list, optional): columns to convert to
            datetime. Defaults to None.
        add_cols (list, optional):
            (new_col_name,new_col_value,cast_to_type) columns to add
            to the dataframe. Defaults to None.
        cols_to_delete (list, optional): columns to delete from
            the dataframe. Defaults to None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        values_to_delete: Optional[List[Tuple[str, str]]] = None,
        datetime_cols: Optional[List[str]] = None,
        add_cols: Optional[List[Tuple[str, str, str]]] = None,
        cols_to_delete: Optional[List[str]] = None,
    ):
        self.df = df
        self.target_col = target_col
        self.datetime_cols = datetime_cols
        self.values_to_delete = values_to_delete
        self.add_cols = add_cols
        self.cols_to_delete = cols_to_delete

    def delete_entries(
        self, df: pd.DataFrame, values_to_delete: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Delete entries from the dataframe

        Args:
            df (pd.DataFrame): input dataframe
            values_to_delete (list): list of tuples (col_name,value to delete)

        Returns:
            pd.DataFrame: dataframe without the specified values
        """
        for col, value in values_to_delete:
            df = df[df[col] != value]
        return df

    def add_entries(
        self, df: pd.DataFrame, add_cols: List[Tuple[str, str, str]]
    ) -> pd.DataFrame:
        """Add new colums to the dataframe with new values.
        These values are combinations of existing columns.

        Args:
            df (pd.DataFrame): input dataframe
            add_cols (list): list of tuples
                (new_col_name,new_col_value,cast_to_type)

        Returns:
            pd.DataFrame: dataframe with new columns
        """
        for new_col_name, new_col_value, cast_to_type in add_cols:
            df[new_col_name] = new_col_value
            if cast_to_type is not None:
                df[new_col_name] = df[new_col_name].astype(cast_to_type)
        return df

    def convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to datetime

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: dataframe with datetime columns
        """
        for col in self.datetime_cols:
            df[col] = pd.to_datetime(df[col])
        return df

    def add_date_infos(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add year, month, day and hour to the dataframe

        Args:
            df (pd.DataFrame): dataframe
            col (str): column name

        Returns:
            pd.DataFrame: dataframe with new columns
        """
        for attr in ["year", "month", "day", "hour"]:
            if getattr(df[col].dt, attr).isnull().sum() == 0:
                df[col + "_" + attr] = getattr(df[col].dt, attr)
        return df

    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data

        Returns:
            pd.DataFrame: dataframe
        """
        df = self.df
        if self.datetime_cols is not None:
            df = self.convert_to_datetime(df)
            for col in self.datetime_cols:
                df = self.add_date_infos(df, col)
                df = df.drop(col, axis=1)
        if self.add_cols is not None:
            df = self.add_entries(df, self.add_cols)
        if self.values_to_delete is not None:
            df = self.delete_entries(df, self.values_to_delete)
        df.dropna(inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        if self.cols_to_delete is not None:
            df = df.drop(self.cols_to_delete, axis=1)
        return df

    def save_cleaned_data(self, df: pd.DataFrame, path: str) -> None:
        """Save the cleaned data

        Args:
            df (pd.DataFrame): dataframe
            path (str): path to save the dataframe
        """
        path = get_path(path, check_dir=True)
        df.to_csv(path, index=False)


class DataCleaner:
    """Clean the data by removing correlated features

    Attributes:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name
        corr_threshold (float): correlation threshold

    Methods:
        compute_correlation_matrix: compute the correlation matrix
        plot_corr_matrix: plot the correlation matrix
        remove_correlated_features: remove correlated features from the
            dataframe with a threshold
        clean_data: clean the data

    Args:
        df (pd.DataFrame): input dataframe
        target_col (str): target column name
        corr_threshold (float, optional): correlation threshold.
            Defaults to 0.7.
    """

    def __init__(self, df, target_col: str, corr_threshold: float = 0.7):
        self.df = df
        self.corr_threshold = corr_threshold
        self.target_col = target_col

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the correlation matrix

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: correlation matrix
        """
        return df.corr().abs()

    def plot_corr_matrix(self, df: pd.DataFrame) -> None:
        """Plot the correlation matrix

        Args:
            df (pd.DataFrame): dataframe
        """
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True)
        plt.show()

    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features from the dataframe with a threshold

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: dataframe without correlated features
        """
        df_copy = df.drop(self.target_col, axis=1)
        corr_matrix = self.compute_correlation_matrix(df_copy)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.corr_threshold)
        ]
        df_copy = df_copy.drop(df[to_drop], axis=1)
        df_copy[self.target_col] = df[self.target_col]
        return df_copy

    def clean_data(self) -> pd.DataFrame:
        """Clean the data

        Returns:
            pd.DataFrame: dataframe
        """
        df = self.remove_correlated_features(self.df)
        return df


def fast_load(
    config_path: str,
    values_to_delete: Optional[List[Tuple[str, str]]] = None,
    adding_cols: Optional[List[Tuple[str, str, str]]] = None,
    keep_corr_features: bool = True,
) -> List:
    """Provides a fast way to load data and preprocess it

    Args:
        config_path (str): path to the config file
        values_to_delete (list, optional): list of tuples
            (col_name,value to delete). Defaults to None.
        adding_cols (list, optional): list of tuples
            (col_name,fun_to_add,cast_to_type). Defaults to None.
        keep_corr_features (bool, optional): whether to keep correlated
            features or not. Defaults to True.

    Returns:
        list: a list containing the data, the target column name,
            the task and the data_cleaner object
    """
    config_path = get_path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)

    path = config["path"]
    target_col = config["target_col"]
    task = config["task"]
    cleaned_data_path = config["cleaned_data_path"]
    if "datetime_cols" not in config.keys():
        datetime_cols = None
    else:
        datetime_cols = config["datetime_cols"]
    if "cols_to_delete" not in config.keys():
        cols_to_delete = None
    else:
        cols_to_delete = config["cols_to_delete"]

    data_loader = LoadData(path=path)
    data = data_loader.load_csv()

    if adding_cols is not None:
        add_cols_copy = adding_cols.copy()
        add_cols_copy = [
            (col_name, fun(data), cast_to_type)
            for col_name, fun, cast_to_type in adding_cols
        ]
    else:
        add_cols_copy = None

    preprocessor = Preprocessor(
        data,
        target_col,
        values_to_delete,
        datetime_cols,
        add_cols_copy,
        cols_to_delete,
    )
    data = preprocessor.preprocess()
    if not keep_corr_features:
        data_cleaner = DataCleaner(data, target_col)
        data = data_cleaner.clean_data()
    else:
        data_cleaner = None
    preprocessor.save_cleaned_data(data, cleaned_data_path)
    return {
        "data": data,
        "target_col": target_col,
        "task": task,
        "data_cleaner": data_cleaner,
    }.values()


def load_data(
    from_cleaned: bool,
    config_path: str,
    values_to_delete: Optional[List[Tuple[str, str]]] = None,
    add_list: Optional[List[Tuple[str, str, str]]] = None,
    keep_corr_features: bool = True,
) -> List:
    """Load data from a config file

    Args:
        from_cleaned (bool): whether to load the data directly from the
            cleaned data or not
        config_path (str): path to the config file
        values_to_delete (list, optional): list of tuples
            (col_name,value to delete). Defaults to None.
        add_list (list, optional): list of tuples
            (col_name,fun_to_add,cast_to_type). Defaults to None.
        keep_corr_features (bool, optional): whether to keep correlated
            features or not. Defaults to True.

    Returns:
        list: a list containing the data, the target column name,
            the task and the data_cleaner object
    """
    config_path = get_path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)
    if from_cleaned:
        path = get_path(config["cleaned_data_path"])
        data = pd.read_csv(path)
        return {
            "data": data,
            "target_col": config["target_col"],
            "task": config["task"],
            "data_cleaner": None,
        }.values()
    data, target_col, task, data_cleaner = fast_load(
        config_path, values_to_delete, add_list, keep_corr_features
    )
    return {
        "data": data,
        "target_col": target_col,
        "task": task,
        "data_cleaner": data_cleaner,
    }.values()
