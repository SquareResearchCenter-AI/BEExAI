import numpy as np
import pandas as pd
import torch


def convert_to_tensor(x, device="cpu"):
    """Convert a numpy array or a pandas dataframe to a torch tensor"""
    if isinstance(x, np.ndarray):
        res = torch.from_numpy(x).to(device)
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        res = torch.from_numpy(x.values).to(device)
    else:
        res = x
    return res


def convert_to_numpy(x):
    """Convert a torch tensor or a pandas dataframe to a numpy array"""
    if isinstance(x, torch.Tensor):
        res = x.detach().cpu().numpy()
    elif isinstance(x, pd.DataFrame):
        res = x.values
    else:
        res = x
    return res
