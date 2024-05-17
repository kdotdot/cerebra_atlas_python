"""Cache class atributes to disk or RAM"""


 


    

import os
import logging
import pickle
from functools import wraps
from typing import Any, Callable, Concatenate, ParamSpec, Tuple, TypeVar
import numpy as np
import mne


logger = logging.getLogger(__name__)


def pkl_load(path):
    """Fn for loading a pickle object"""
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res


def pkl_save(path, result):
    """Fn for saving as a pickle object"""
    with open(path, "wb") as f:
        res = pickle.dump(result, f)
    return res


def cache_pkl():
    """Cache as pkl object"""
    return _cache(pkl_load, pkl_save)


def property_test() -> Callable[..., Callable[..., Any]]:
    def decorator(func: Callable[..., Tuple[Callable[..., np.ndarray], str | None]]):
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> np.ndarray:
            compute_fn, str = func(self, *args, **kwargs)
            return compute_fn()

        return wrapper

    return decorator


def cache_np(
    compute_fn: Callable[..., np.ndarray], cached_path: str, *args, **kwargs
) -> np.ndarray:
    """Cache numpy array to disk"""

    def load_data_fn(path: str) -> np.ndarray:
        return np.load(path)

    def save_data_fn(path: str, result: np.ndarray) -> None:
        return np.save(path, result)

    return _cache(compute_fn, cached_path, load_data_fn, save_data_fn)


def cache_mne_src():
    """Cache mne src space to disk"""

    def load_data_fn(path):
        return mne.read_source_spaces(path)

    def save_data_fn(path, result):
        return result.save(path, overwrite=True, verbose=False)

    return _cache(load_data_fn, save_data_fn)


def cache_mne_bem():
    """Cache mne BEM to disk"""

    def load_data_fn(path):
        return mne.read_bem_solution(path, verbose=False)

    def save_data_fn(path, result):
        return mne.write_bem_solution(path, result, overwrite=True, verbose=True)

    return _cache(load_data_fn, save_data_fn)


def cache_attr() -> Callable[..., Any]:
    """Cache attribute in RAM. Do not save to disk"""
    return _cache(None, None)


T = TypeVar("T")
type LoadDataFn = Callable[[str], T] | None
type SaveDataFn = Callable[[str, T], None] | None
type ComputeFn = Callable[..., T]
type CachePath = str | None

def _cache(
    compute_fn: Callable[..., T],
    cached_path,
    load_data_fn,
    save_data_fn: Callable[[str, T], None] | None = None,
) -> T:

    def decorator(func: Callable[..., Tuple[Callable[..., T], str | None]]) -> T:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            property_name = f"_{func.__name__}"

            if not hasattr(self, property_name):
                setattr(self, property_name, None)
            # If property already exists, return it
            if (
                hasattr(self, property_name)
                and getattr(self, property_name) is not None
            ):
                logger.debug("Property already exists")
                return getattr(self, property_name)

            # Otherwise compute the result
            compute_fn, save_path = func(self, *args, **kwargs)

            if save_path is None:
                result = compute_fn(self)
                setattr(self, property_name, result)
                return result

            assert save_data_fn is not None and load_data_fn is not None

            # If property is None, try to load from disk
            if os.path.exists(save_path):
                logger.debug("Loading cached result from %s", save_path)
                result = load_data_fn(save_path)
                setattr(self, property_name, result)
                return result

            # Compute the result
            result = compute_fn(self)

            # Save to disk if needed
            if save_path is not None:

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_data_fn(save_path, result)
                logger.debug("Saved result to %s", save_path)

            # Set the property and return the result
            setattr(self, property_name, result)
            return result

        return wrapper

    return decorator
