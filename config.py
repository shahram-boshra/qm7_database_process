# config.py

import yaml
from pathlib import Path
from typing import Any, Dict, Union, Type, Optional

try:
    from exceptions import (
        ConfigNotFound,
        InvalidConfigError,
        ConfigAttributeMissingError,
        ConfigTypeError
    )
except ImportError:
    class ConfigNotFound(FileNotFoundError):  
        """Exception raised when a configuration file is not found."""
        def __init__(self, path: Path, message: str = "Configuration file not found", original_exception: Optional[Exception] = None) -> None:
            self.path = path
            self.original_exception = original_exception
            super().__init__(f"{message}: '{path}'")

    class InvalidConfigError(ValueError):  
        """Exception raised for issues with the configuration file content or format."""
        def __init__(self, path: Optional[Path] = None, message: str = "Invalid configuration", original_exception: Optional[Exception] = None) -> None:
            self.path = path
            self.original_exception = original_exception
            if path:
                super().__init__(f"{message} in file: '{path}'")
            else:
                super().__init__(message)

    class ConfigAttributeMissingError(AttributeError):  
        """Exception raised when a required configuration attribute is missing."""
        def __init__(self, attribute_path: str, message: str = "Configuration attribute missing") -> None:
            self.attribute_path = attribute_path
            super().__init__(f"{message}: '{attribute_path}'")

    class ConfigTypeError(TypeError):  
        """Exception raised when a configuration value has an unexpected type."""
        def __init__(self, key_path: str, expected_type: Type[Any], actual_type: Type[Any], value: Any, message: str = "Incorrect type for configuration attribute") -> None:
            self.key_path = key_path
            self.expected_type = expected_type
            self.actual_type = actual_type
            self.value = value
            super().__init__(f"{message} at '{key_path}'. Expected '{expected_type.__name__}', got '{actual_type.__name__}' for value '{value}'.")


class Config:
    """
    A class to load and access configuration parameters from a YAML file.

    It provides dot-notation access to configuration settings, including nested
    structures, and includes robust error handling for file operations, YAML
    parsing, and attribute access.
    """
    def __init__(self, config_file_path: Path) -> None:
        """
        Initializes the Config object by loading and parsing the specified YAML file.

        Args:
            config_file_path (Path): The path to the YAML configuration file.

        Raises:
            TypeError: If `config_file_path` is not a `Path` object.
            ConfigNotFound: If the specified configuration file does not exist.
            InvalidConfigError: If the YAML file is malformed or its root is not a dictionary.
            ConfigAttributeMissingError: If any key in the configuration is not a valid Python identifier.
        """
        if not isinstance(config_file_path, Path):
            raise TypeError(f"config_file_path must be a Path object, got {type(config_file_path)}")

        self._config_data: Dict[str, Any] = self._load_yaml(config_file_path)
        if not isinstance(self._config_data, dict):
            raise InvalidConfigError(
                path=config_file_path,
                message=f"Root of configuration file '{config_file_path}' must be a dictionary, but got {type(self._config_data).__name__}."
            )
        self._set_attributes(self._config_data)

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Loads YAML file content with specific error handling.

        Args:
            file_path (Path): The path to the YAML file.

        Returns:
            Dict[str, Any]: The loaded YAML content as a dictionary.

        Raises:
            ConfigNotFound: If the file does not exist.
            InvalidConfigError: If there's an error parsing the YAML or any other unexpected error during loading.
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ConfigNotFound(path=file_path, original_exception=e) from e
        except yaml.YAMLError as e:
            raise InvalidConfigError(
                path=file_path,
                message=f"Error parsing configuration file '{file_path}': {e}",
                original_exception=e
            ) from e
        except Exception as e:
            raise InvalidConfigError(
                path=file_path,
                message=f"An unexpected error occurred while loading '{file_path}': {e}",
                original_exception=e
            ) from e

    def _set_attributes(self, data: Dict[str, Any], parent_key: str = '') -> None:
        """
        Recursively sets attributes for nested dictionaries, handling potential non-dict data at root.

        This method iterates through the provided dictionary `data` and sets each
        key-value pair as an attribute of the `Config` instance. If a value is
        itself a dictionary, it creates a `ConfigNode` instance for it, allowing
        further dot-notation access.

        Args:
            data (Dict[str, Any]): The dictionary containing configuration data.
            parent_key (str, optional): The concatenated path of parent keys for
                                        error reporting in nested structures. Defaults to ''.

        Raises:
            InvalidConfigError: If `data` (or a nested part of it) is expected to be a dictionary but isn't.
            ConfigAttributeMissingError: If a key is not a valid Python identifier or if
                                        an error occurs during `ConfigNode` instantiation.
        """
        if not isinstance(data, dict):
            raise InvalidConfigError(
                message=f"Configuration data at path '{parent_key}' must be a dictionary, but got {type(data)}."
            )

        for key, value in data.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ConfigAttributeMissingError(
                    attribute_path=f"{parent_key}.{key}" if parent_key else key,
                    message=f"Invalid key '{key}' found in configuration at path '{parent_key}'. Keys must be valid Python identifiers."
                )

            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                try:
                    setattr(self, key, ConfigNode(value, full_key))
                except Exception as e:
                    raise ConfigAttributeMissingError(
                        attribute_path=full_key,
                        message=f"Error processing nested dictionary for key '{full_key}': {e}",
                        original_exception=e
                    ) from e
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Custom `__getattr__` to provide more informative error messages for missing attributes.

        If an attribute is accessed that does not exist directly on the `Config`
        instance, this method is called. It raises a `ConfigAttributeMissingError`
        to indicate that the requested configuration key was not found.

        Args:
            name (str): The name of the attribute being accessed.

        Raises:
            ConfigAttributeMissingError: Always, as it means the configuration key was not found.
        """
        raise ConfigAttributeMissingError(f"Configuration key '{name}' not found.")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Config object, primarily showing its underlying data.
        """
        return f"Config({self._config_data})"


class ConfigNode:
    """
    Helper class to allow dot-notation access to nested config dictionaries with enhanced error handling.

    This class is used internally by the `Config` class to represent nested
    dictionary structures, enabling intuitive access to configuration values
    (e.g., `cfg.section.subsection.item`). It also provides methods for
    safe retrieval and type validation of values within a node.
    """

    def __init__(self, data: Dict[str, Any], path: str = '') -> None:
        """
        Initializes a ConfigNode with a given dictionary and its path within the configuration.

        Args:
            data (Dict[str, Any]): The dictionary representing the current level of configuration.
            path (str, optional): The dot-separated path to this node from the root of the config. Defaults to ''.

        Raises:
            InvalidConfigError: If the provided `data` is not a dictionary.
            ConfigAttributeMissingError: If a key within `data` is not a valid Python identifier or
                                        if an error occurs during recursive `ConfigNode` instantiation.
        """
        if not isinstance(data, dict):
            raise InvalidConfigError(
                path=path,
                message=f"ConfigNode data must be a dictionary, got {type(data).__name__} for path '{path}'."
            )
        self._data = data
        self._path = path
        for key, value in data.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ConfigAttributeMissingError(
                    attribute_path=f"{path}.{key}" if path else key,
                    message=f"Invalid key '{key}' found at path '{path}'. Keys must be valid Python identifiers."
                )

            full_key = f"{self._path}.{key}" if self._path else key
            if isinstance(value, dict):
                try:
                    setattr(self, key, ConfigNode(value, full_key))
                except Exception as e:
                    raise ConfigAttributeMissingError(
                        attribute_path=full_key,
                        message=f"Error processing nested dictionary for key '{full_key}': {e}",
                        original_exception=e
                    ) from e
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Safely gets a value associated with the given key from the node's data.

        If the key is not found, it returns the specified default value instead of raising an error.

        Args:
            key (str): The key to retrieve.
            default (Optional[Any], optional): The value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if the key is not present.
        """
        return self._data.get(key, default)

    def get_typed(self, key: str, expected_type: Type[Any], default: Optional[Any] = None) -> Any:
        """
        Gets a value for the given key and validates its type.

        If the value is not `None` and does not match the `expected_type`, a
        `ConfigTypeError` is raised.

        Args:
            key (str): The key to retrieve.
            expected_type (Type[Any]): The expected Python type of the value.
            default (Optional[Any], optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if the key is not present,
                 after type validation.

        Raises:
            ConfigTypeError: If the retrieved value is not `None` and its type does not match `expected_type`.
        """
        value = self._data.get(key, default)
        if value is not None and not isinstance(value, expected_type):
            raise ConfigTypeError(
                key_path=f"{self._path}.{key}",
                expected_type=expected_type,
                actual_type=type(value),
                value=value
            )
        return value

    def __getitem__(self, key: str) -> Any:
        """
        Allows dictionary-style access (e.g., `node['key']`) with more informative error for missing keys.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            ConfigAttributeMissingError: If the key is not found in the node's data.
        """
        try:
            return self._data[key]
        except KeyError:
            raise ConfigAttributeMissingError(f"Configuration key '{self._path}.{key}' not found.")

    def __getattr__(self, name: str) -> Any:
        """
        Custom `__getattr__` for ConfigNode to provide more informative error messages for missing attributes.

        This method is called when an attribute is accessed via dot-notation (e.g., `node.key`)
        that is not directly defined on the `ConfigNode` instance. It attempts to retrieve
        the value from the underlying dictionary `_data`. If the key is not found, it raises
        a `ConfigAttributeMissingError`.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value associated with the key from the underlying data.

        Raises:
            ConfigAttributeMissingError: If the configuration key is not found.
        """
        if name in self._data:
            return self._data[name]
        raise ConfigAttributeMissingError(f"Configuration key '{self._path}.{name}' not found.")

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key exists within the `ConfigNode`'s data.

        Args:
            key (str): The key to check for existence.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self._data

    def __repr__(self) -> str:
        """
        Returns a string representation of the ConfigNode object, primarily showing its underlying data.
        """
        return f"ConfigNode({self._data})"


if __name__ == "__main__":
    dummy_config_content = """
    pipeline:
      base_data_dir: "/tmp/Chem_Data_Test"
    download:
      qm7_sdf_url: "http://example.com/gdb7.tar.gz"
    processing:
      chunk_size: 100
      pre_filters:
        enable_max_nodes_filter: true
        max_nodes: 20
        enable_min_nodes_filter: false
        min_nodes: 2
        enable_num_carbons_filter: true
        min_carbons: 1
    curation:
      feature_keys_for_normalization: ["x", "y"]
      additional_pyg_transforms:
        - name: "ToUndirected"
        - name: "AddSelfLoops"
          kwargs:
            loop_type: "self"
    output_files:
      curated_dataset_name: "curated_data_test.pt"
    """
    dummy_config_path = Path(__file__).resolve().parent / "dummy_config.yaml"
    with open(dummy_config_path, 'w') as f:
        f.write(dummy_config_content)

    print(f"Loading config from: {dummy_config_path}")
    try:
        cfg = Config(dummy_config_path)
        print("\nConfig loaded successfully:")
        print(f"Base Data Dir: {cfg.pipeline.base_data_dir}")
        print(f"QM7 SDF URL: {cfg.download.qm7_sdf_url}")
        print(f"Chunk Size: {cfg.processing.chunk_size}")

        enable_max_nodes = cfg.processing.pre_filters.get_typed("enable_max_nodes_filter", bool)
        print(f"Enable Max Nodes Filter (typed): {enable_max_nodes}")

        print(f"Max Nodes: {cfg.processing.pre_filters.max_nodes}")
        print(f"Feature Keys for Norm: {cfg.curation.feature_keys_for_normalization}")
        print(f"First Additional Transform Name: {cfg.curation.additional_pyg_transforms[0]['name']}")
        print(f"Second Additional Transform Name: {cfg.curation.additional_pyg_transforms[1]['name']}")
        print(f"Second Additional Transform Kwargs: {cfg.curation.additional_pyg_transforms[1]['kwargs']}")
        print(f"Curated Dataset Name: {cfg.output_files.curated_dataset_name}")

        print("\n--- Testing Error Handling ---")

        try:
            print(cfg.non_existent_section)
        except ConfigAttributeMissingError as e:
            print(f"Caught expected error (non-existent top-level key): {e}")

        try:
            print(cfg.pipeline.non_existent_item)
        except ConfigAttributeMissingError as e:
            print(f"Caught expected error (non-existent nested key): {e}")

        try:
            print(cfg.curation.additional_pyg_transforms[99]['name'])
        except IndexError as e:
            print(f"Caught expected error (list index out of range): {e}")

        try:
            print(cfg.curation.additional_pyg_transforms[0]['non_existent_prop'])
        except ConfigAttributeMissingError as e:
            print(f"Caught expected error (non-existent key in dict in list): {e}")

        malformed_yaml_path = Path(__file__).resolve().parent / "malformed_config.yaml"
        with open(malformed_yaml_path, 'w') as f:
            f.write("key: [malformed\n  value: 123")
        try:
            _ = Config(malformed_yaml_path)
        except InvalidConfigError as e:
            print(f"Caught expected error (malformed YAML): {e}")
        finally:
            if malformed_yaml_path.exists():
                malformed_yaml_path.unlink()

        try:
            _ = Config(Path("non_existent_file.yaml"))
        except ConfigNotFound as e:
            print(f"Caught expected error (file not found): {e}")

        print("\n--- Testing Type Validation ---")
        try:
            cfg.processing.pre_filters._data['enable_max_nodes_filter'] = "true"
            bad_type_value = cfg.processing.pre_filters.get_typed("enable_max_nodes_filter", bool)
            print(
                f"Enable Max Nodes Filter (bad type, but got): {bad_type_value}")
        except ConfigTypeError as e:
            print(f"Caught expected error (incorrect type): {e}")

        malformed_key_path = Path(__file__).resolve().parent / "malformed_key_config.yaml"
        with open(malformed_key_path, 'w') as f:
            f.write("'bad key': value")
        try:
            _ = Config(malformed_key_path)
        except ConfigAttributeMissingError as e:
            print(f"Caught expected error (invalid key name): {e}")
        finally:
            if malformed_key_path.exists():
                malformed_key_path.unlink()

        print("\n--- Testing Init with non-Path object ---")
        try:
            _ = Config("not_a_path.yaml")  # type: ignore
        except TypeError as e:
            print(f"Caught expected error (non-Path init): {e}")


    except Exception as e:
        print(f"An unexpected error occurred during config loading test: {e}")
    finally:
        if dummy_config_path.exists():
            dummy_config_path.unlink()
