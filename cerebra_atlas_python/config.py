import logging
from abc import ABC, abstractmethod
from cerebra_atlas_python.utils import read_config_as_dict

# 4 options:
# [1]: Call MNIAverage without parameters & config file exists
# mniAverage = MNIAverage() -> Use config
# [2]: Call MNIAverage without parameters & config file does not exist
# mniAverage = MNIAverage() -> Use MNIAverage default config
# [3]: Call MNIAverage with parameters & config file exists
# mniAverage = MNIAverage(bem_ico=3) -> Use first parameters, then config
# [4]: Call MNIAverage with parameters & config file does not exist
# mniAverage = MNIAverage(bem_ico=3) -> Use first parameters, then MNIAverage default config


class BaseConfig(ABC):  # Abstract class
    @abstractmethod
    def __init__(
        self, parent_name: str, default_config: (dict or None) = None, **kwargs
    ):
        """
        Purpose:
        BaseConfig serves as an abstract base class for configuration management. It is designed to:

        1. Read configuration settings from a file.
        2. Override these settings with default values if the file is not available or lacks certain settings.
        3. Allow further customization through runtime parameters.

        Usage:
        This class is intended to be subclassed by other classes that require configurable settings.
        The subclass should provide its own parent_name and default_config as appropriate.

        Args:
            parent_name (str): Name of the parent class
            default_config (dict or None, optional):  Default configuration values. Used if the configuration file is missing or incomplete. Defaults to None.
            **kwargs: Additional keyword arguments to override both the configuration file and default values.
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
                    f"Config and default values were not provided for class {parent_name}"
                )

        # Update missing keys in the config with default values
        for key, value in default_config.items():
            if key not in config:
                logging.warning(
                    f"Value for variable {key} not provided in {parent_name}. Defaulting to {key}={value}"
                )
            config.setdefault(key, value)

        # Override with any provided kwargs
        config.update(kwargs)

        # Set each configuration item as an instance attribute
        for key, value in config.items():
            try:
                # Attempt to evaluate the value if it's a string that represents a Python literal
                parsed_value = eval(value)
            except:
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
