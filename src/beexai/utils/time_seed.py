"""Utility functions"""

import random
import time
from functools import wraps
from typing import Callable

import numpy as np
import torch


def time_function(func: Callable) -> Callable:
    """Decorator to time a function

    Args:
        func (callable): function to time

    Returns:
        callable: timed function
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    _wrapper.__name__ = func.__name__
    _wrapper.__doc__ = func.__doc__
    return _wrapper


def set_seed(seed: int = 42) -> None:
    """Set the random seed

    Args:
        seed (int, optional): seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
