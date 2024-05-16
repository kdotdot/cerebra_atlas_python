"""Caching utils"""

import logging
import os
import os.path as op
from typing import Callable, Dict, TypeVar
import pickle
import inspect
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


def cache_np(
    compute_fn: Callable[..., np.ndarray], cached_path: str, *args, **kwargs
) -> np.ndarray:
    """Cache numpy array to disk. If cached path exists, return it. Otherwise compute the result and save to disk.


    Args:
        compute_fn (Callable[..., np.ndarray]): "Expensive" function to compute the result. Not called if cached path exists.
        cached_path (str): path to cached file

    Returns:
        np.ndarray: result of compute_fn
    """

    def load_data_fn(path: str) -> np.ndarray:
        return np.load(path)

    def save_data_fn(path: str, result: np.ndarray) -> None:
        return np.save(path, result)

    return _cache(compute_fn, cached_path, load_data_fn, save_data_fn, *args, **kwargs)


def cache_pkl(
    compute_fn: Callable[..., Dict[int, np.ndarray]], cached_path: str, *args, **kwargs
) -> Dict[int, np.ndarray]:
    """Cache pkl object to disk. If cached path exists, return it. Otherwise compute the result and save to disk."""

    def pkl_load(path):
        with open(path, "rb") as f:
            res = pickle.load(f)
        return res

    def pkl_save(path, result):
        with open(path, "wb") as f:
            pickle.dump(result, f)

    return _cache(compute_fn, cached_path, pkl_load, pkl_save, *args, **kwargs)


T = TypeVar("T")

# "type" kw requires Python >=3.12
# type LoadDataFn = Callable[[str], T] | None
# type SaveDataFn = Callable[[str, T], None] | None
# type ComputeFn = Callable[..., T]
# type CachePath = str | None


def _add_fn_hash_to_path(fn: Callable, path: str):
    path_split = path.split(".")
    source_code = inspect.getsource(fn)
    fn_name = fn.__name__
    defaults = fn.__defaults__
    combined = f"{fn_name}{source_code}{defaults}"
    fn_hash = hashlib.sha256(combined.encode()).hexdigest()
    path = ".".join(path_split[:-1]) + f"_{fn_hash}." + path_split[-1]
    return path


def _cache(
    compute_fn: Callable[..., T],
    cached_path: str,
    load_data_fn: Callable[..., T],
    save_data_fn: Callable[[str, T], None],
    *args,
    **kwargs,
) -> T:
    """Helper function to cache data to disk. If cached path exists, return it. Otherwise compute the result and save to disk.
    Use hash to ensure that the cached path is unique to the function and its arguments.
    """
    # Add hash to name
    cached_path = _add_fn_hash_to_path(compute_fn, cached_path)
    # If cached path exists, return it
    if op.exists(cached_path):
        logger.debug("Loading cached result from %s", cached_path)
        return load_data_fn(cached_path)
    # Otherwise compute the result
    result = compute_fn(*args, **kwargs)
    # Save to disk if needed
    os.makedirs(os.path.dirname(cached_path), exist_ok=True)

    save_data_fn(cached_path, result)
    logger.debug("Saved result to %s", cached_path)
    return result
