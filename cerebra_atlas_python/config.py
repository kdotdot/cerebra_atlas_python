"""
This module provides a class for configuring class attributes dinamically
"""
import os
import os.path as op
import logging
from typing import Tuple, Dict, Optional, Any
from configparser import ConfigParser, InterpolationMissingOptionError
from abc import ABC

logger = logging.getLogger(__name__)

def read_config_as_dict(
    file_path: str, section: Optional[str] = None
) -> Tuple[Dict[str, str], bool]:
    """
    Reads a configuration file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the configuration file.
        section (Optional[str]): Specific section to read from the configuration file.
                                    If None, all sections are read. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, str]], bool]: A tuple containing a dictionary of configuration values
                                                and a boolean indicating success.
    """

    # ALLOW MISSING ENV VARIABLES (env variables which are empty throw InterpolationMissingOptionError)
    def attempt_get_value(parser: ConfigParser, section: str, key: str) -> str:
        try:
            return parser[section][key]
        except InterpolationMissingOptionError:
            return ""

    config_dict: Dict[str, Dict[str, Any]] = {}
    success: bool = True

    if file_path is None or not op.exists(file_path):
        if file_path is not None:
            logging.warning("Config file does not exist: %s", file_path)
        success = False
        return config_dict, success


    default_env = {env_k:env_v for env_k, env_v in os.environ.items() if ("%" not in env_k and "%" not in env_v)  }

    config_parser = ConfigParser()
    config_parser.read_dict({"DEFAULT": default_env})
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

    return config_dict, success

def is_valid_config_file(path):
    if not path.endswith(".ini"):
        logging.debug("False, Config file must be a .ini file %s", path)
        return False
    if not op.exists(path):
        logging.debug("False, Config file does not exist: %s", path)
        return False
    return True



def search_config_path(raise_on_missing_config_file: bool = False):    
    config_path = None
    # Use CONFIG env variable if set
    if "CONFIG" in os.environ:
        config_path = os.environ["CONFIG"]
        if is_valid_config_file(config_path):
            logger.debug("Using config file from env variable CONFIG %s", config_path)
            return config_path

    # If env is not set, use the first .ini found in current working dir
    cwd = os.getcwd()
    config_file_name = None
    for fname in os.listdir(cwd):
        if fname.endswith(".ini"):
            if config_file_name is None:
                config_file_name = fname
            else:
                logging.warning(
                    "Multiple .ini files found in working directory. Using %s", config_file_name
                )
    if config_file_name is not None:
        config_path = op.join(cwd, config_file_name)    
        if is_valid_config_file(config_path):    
            logger.debug("Using config %s from current working dir %s", config_file_name, cwd)
            return config_path
    logger.debug("Config file not found in current working dir %s", cwd)

    # If no ini is found, use the default config.ini for the package 
    config_path = op.dirname(__file__) + "/config.ini"
    if is_valid_config_file(config_path):
            logger.debug("Using default config from module's path %s", config_path)
            return config_path

    if raise_on_missing_config_file:
        raise FileNotFoundError("No valid config file found")

    logger.warning("No config file was found")

    return None

def search_and_get_config():
    config_path = search_config_path()
    if config_path is not None:
        config, success = read_config_as_dict(file_path=config_path)
        return config if success else {}

    return {}

# IF NO CONFIG FILE IS FOUND, AND NO KWARGS ARE PROVIDED, DEFAULTS ARE USED
# IF A CONFIG FILE IS FOUND, IT OVERRIDES DEFAULTS FOR PRESENT FIELDS
# IF KWARGS ARE PROVIDED, OVERRIDE EVERYTHING ELSE

# CONFIG FILES ARE REQUIRED!
class Config(ABC):
    def __init__(
        self,class_name: str, config_path: Optional[str] = None, **kwargs):
        if config_path is None:
            config_path = search_config_path(raise_on_missing_config_file=True)

        # Attempt to read configuration
        config, config_success = read_config_as_dict(file_path=config_path, section=class_name)
        if not config_success:
            raise FileNotFoundError(f"Config file {config_path} not found for class {class_name}")
        
        assert config_success, f"Config file {config_path} not found for class {class_name}"

        self.config_path = config_path

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

if __name__ == "__main__":
    config_path = search_config_path(raise_on_missing_config_file=True)
    read_config_as_dict(file_path=config_path, section="HBNDataset")