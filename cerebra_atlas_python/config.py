"""
This module provides a class for configuring class attributes dinamically
"""
import os
import os.path as op
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
from configparser import ConfigParser, InterpolationMissingOptionError


def read_config_as_dict(
    file_path: str = op.dirname(__file__) + "/config.ini", section: Optional[str] = None
) -> Tuple[Dict[str, str], bool]:
    """
    Reads a configuration file and returns its contents as a dictionary.

    This function reads the specified configuration file and parses its contents.
    If a specific section is requested, only that section is returned. Otherwise,
    all sections are returned. Additionally, environment variables are used as
    default values.

    Args:
        file_path (str): Path to the configuration file. Defaults to 'config.ini' in the current directory.
        section (Optional[str]): Specific section to read from the configuration file.
                                    If None, all sections are read. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, str]], bool]: A tuple containing a dictionary of configuration values
                                                and a boolean indicating the success of reading the file.
                                                The dictionary is structured with sections as keys and
                                                dictionaries of the section's key-value pairs as values.
    """

    # ALLOW MISSING ENV VARIABLES (env variables which are empty throw InterpolationMissingOptionError)
    def attempt_get_value(parser: ConfigParser, section: str, key: str) -> str:
        try:
            return parser[section][key]
        except InterpolationMissingOptionError:
            return ""

    config_dict: Dict[str, Dict[str, Any]] = {}
    success: bool = True

    if not op.exists(file_path):
        logging.warning("Config file does not exist: %s", file_path)
        success = False
        return config_dict, success

    config_parser = ConfigParser()
    config_parser.read_dict({"DEFAULT": os.environ})
    config_parser.read(file_path)

    if section is None:
        for section in config_parser.sections():
            config_dict[section] = {}
            # Adds default values to every section
            for key in [*config_parser[section], *config_parser["DEFAULT"]]:
                if key.upper() not in os.environ:
                    config_dict[section][key] = attempt_get_value(
                        config_parser, section, key
                    )

        if not config_dict:
            logging.warning("Attempt to read empty config file: %s", file_path)
            success = False
    else:
        if section not in config_parser.sections():
            logging.error(
                "Section '%s' does not exist in config file: %s", section, file_path
            )
            success = False
        else:
            config_dict[section] = {}
            # Adds default values to every section
            for key in [*config_parser[section], *config_parser["DEFAULT"]]:
                if key.upper() not in os.environ:
                    config_dict[section][key] = attempt_get_value(
                        config_parser, section, key
                    )
            config_dict = config_dict[section]

    print(config_dict)

    return config_dict, success


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
        config_path: Optional[str] = None,
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
            config_path (Optional[str]): Path to configuration file
            **kwargs: Additional keyword arguments representing runtime configuration overrides.

        Note:
            The configuration values are set as instance attributes, making them directly accessible
            as properties of the class instance.
        """
        default_config = default_config or {}

        # Attempt to read the configuration
        config, config_success = read_config_as_dict(
            file_path=config_path, section=parent_name
        )

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
            except (
                SyntaxError,
                TypeError,
                NameError,
            ):  # String / Boolean throw error at eval()
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


if __name__ == "__main__":
    read_config_as_dict()
    read_config_as_dict(section="CerebrA")
