import os
import os.path as op
from functools import wraps
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


def cache_pkl():
    def load_data_fn(path):
        with open(path, 'rb') as f:
            res =  pickle.load(f)
        return res
    def save_data_fn(path, result):
        with open(path, 'wb') as f:
            res  =  pickle.dump(result, f)
        return res
    return _cache(load_data_fn, save_data_fn)

def cache_np():
    def load_data_fn(path):
        return np.load(path)
    def save_data_fn(path, result):
        return np.save(path, result)
    return _cache(load_data_fn, save_data_fn)


def cache_mne_src():
    import mne
    def load_data_fn(path):
        mne.read_source_spaces(path)
    def save_data_fn(path, result):
        result.save(path, overwrite=True, verbose=False)
        
    return _cache(load_data_fn, save_data_fn)

def _cache(load_data_fn, save_data_fn):
    """
    Decorator for caching class properties
    Example usage:

    If output_path_property_name is None:

    -------
    @property
    def src_space_mask(self):
        if self._src_space_mask is None:
            self._src_space_mask = self._get_src_space_mask()
        return self._src_space_mask

    is equivalent to 

    @property
    @cache("_src_space_mask")
    def src_space_mask(self):
        return self._get_src_space_mask()
    -------

    If output_path_property_name is generated folder
    Then save to disk:

    -------save_function
    @property
    def src_space_mask(self):
        if self._src_space_mask is None:
            if os.exists(self.cerebra_output_path):
                self._src_space_mask = np.load(self.cerebra_output_path)
            else:
                self._src_space_mask = self._get_src_space_mask()
                np.save(self.cerebra_output_path, self._src_space_mask)
        return self._src_space_mask

    is equivalent to 

    @property
    @cache("_src_space_mask", "cerebra_output_path")
    def src_space_mask(self):
        return self._get_src_space_mask()
    -------

    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            property_name = f"_{func.__name__}"

            if not hasattr(self, property_name):
                setattr(self, property_name, None)
            # If property already exists, return it
            if hasattr(self, property_name) and getattr(self, property_name) is not None:
                logger.debug("Property already exists")
                return getattr(self, property_name)
            

            # Otherwise compute the result
            compute_fn, save_path = func(self, *args, **kwargs)

            # If property is None, try to load from disk
            if os.path.exists(save_path):
                logger.debug(f"Loading cached result from {save_path}")
                result = load_data_fn(save_path)
                setattr(self, property_name, result)
                return result

            # Compute the result
            result = compute_fn(self)

            # Save to disk if needed
            if save_path is not None:

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_data_fn(save_path, result)
                logger.debug(f"Saved result to {save_path}")
            
            # Set the property and return the result
            setattr(self, property_name, result)
            return result
        
        return wrapper
    return decorator