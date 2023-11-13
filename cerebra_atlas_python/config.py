"""
This module provides a class for configuring class attributes dinamically
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from cerebra_atlas_python.utils import read_config_as_dict


class BaseConfig(ABC):  # Abstract class
    """
    An abstract base class for configuration management in Python applications.

    This class serves as a foundation for creating configuration management systems,
    allowing configurations to be read from a file, using default values when necessary,
    and supporting runtime parameter overrides.

    Attributes:
        name (str): The name of the parent class utilizing this configuration.
    """

    @abstractmethod
    def __init__(
        self,
        parent_name: str,
        default_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Constructs a BaseConfig object with specified parameters and default settings.

        Reads configuration from a file specific to the 'parent_name' class. If the file is
        not found or is incomplete, it falls back to the provided default configuration.
        Additional runtime parameters can override both file-based and default configurations.

        Args:
            parent_name (str): The name of the parent class for which the configuration is managed.
            default_config (Optional[Dict[str, Any]]): A dictionary of default configuration values.
                                                      Used if the configuration file is missing or incomplete.
                                                      Defaults to None.
            **kwargs: Additional keyword arguments representing runtime configuration overrides.

        Note:
            The configuration values are set as instance attributes, making them directly accessible
            as properties of the class instance.
        """
        default_config = default_config or {}

        # Attempt to read the configuration
        config, config_success = read_config_as_dict(section=parent_name)

        # Choose config.ini over default_config
        if not config_success:
            # Use the provided default configuration if file reading is unsuccessful
            config = default_config
            if not config:
                logging.warning(
                    "Config and default values were not provided for class %s",
                    parent_name,
                )
        # Update missing keys in the config with default values
        for key, value in default_config.items():
            if key not in config:
                logging.debug(
                    "Value for variable %s not provided in config.ini[%s]. Defaulting to %s=%s",
                    key,
                    parent_name,
                    key,
                    value,
                )
            config.setdefault(key, value)

        # Override with any provided kwargs
        config.update(kwargs)

        # Set each configuration item as an instance attribute
        for key, value in config.items():
            try:
                # Attempt to evaluate the value if it's a string that represents a Python literal
                parsed_value = eval(value)
            except (SyntaxError, TypeError):  # String / Boolean throw error at eval()
                # Retain the original string value if eval fails
                parsed_value = value
            setattr(self, key, parsed_value)
        # Store the parent's name
        self._baseconfig_parent_name = parent_name

    @property
    def name(self) -> str:
        """Returns the name of the parent class.
        Returns:
            str: Name of the parent class
        """
        return self._baseconfig_parent_name
