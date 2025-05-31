# exceptions.py

"""
This module defines custom exception classes for the QM7 data pipeline.
These exceptions provide more specific and granular error handling,
improving the clarity and maintainability of the codebase.
"""

from typing import Optional, Type, Any, List, Dict


class PipelineError(Exception):
    """Base class for exceptions in the QM7 data pipeline."""
    pass


class ImportConfigError(PipelineError):
    """Raised when the Config class from config.py cannot be imported."""

    def __init__(self, message: str = "Failed to import Config class. Ensure config.py is in the correct path and not corrupted.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class ImportModuleError(PipelineError):
    """Raised when a required custom module (e.g., db_dl, pyg_qm7_processing) cannot be imported."""

    def __init__(self, module_name: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to import module '{module_name}'. Ensure {module_name}.py exists and is accessible."
        super().__init__(message)
        self.module_name: str = module_name
        self.original_exception: Optional[Exception] = original_exception


class ImportPyGError(PipelineError):
    """Raised when PyTorch Geometric modules cannot be imported."""

    def __init__(self, message: str = "Failed to import PyTorch Geometric modules. Please ensure torch_geometric is installed.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class ConfigError(PipelineError):
    """Base class for configuration-related errors."""
    pass


class ConfigNotFound(ConfigError):
    """Raised when the configuration file (config.yaml) is not found."""

    def __init__(self, path: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Configuration file not found at: {path}. Please ensure config.yaml exists."
        super().__init__(message)
        self.path: str = path


class ConfigAttributeMissingError(ConfigError):
    """Raised when a required attribute or key is missing from the configuration or is invalid."""

    def __init__(self, attribute_path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Configuration error: '{attribute_path}' is missing or invalid. Please check config.yaml."
        super().__init__(message)
        self.attribute_path: str = attribute_path
        self.original_exception: Optional[Exception] = original_exception


class InvalidConfigError(ConfigError):
    """Raised for general issues with config loading or parsing (e.g., malformed YAML, incorrect root type)."""

    def __init__(self, path: Optional[str] = None, message: str = "Failed to load configuration. Check config.yaml for syntax errors or invalid structure.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.path: Optional[str] = path
        self.original_exception: Optional[Exception] = original_exception


class MissingDownloadURLError(ConfigError):
    """Raised when essential download URLs are missing from the configuration."""

    def __init__(self, url_key: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Configuration error: '{url_key}' is missing. Cannot download data."
        super().__init__(message)
        self.url_key: str = url_key


class ConfigTypeError(ConfigError):
    """Raised when a configuration value has an incorrect type."""

    def __init__(self, key_path: str, expected_type: Type, actual_type: Type, value: Any = None, message: Optional[str] = None) -> None:
        if message is None:
            message = (f"Configuration key '{key_path}' expected type {expected_type.__name__}, "
                       f"but got {actual_type.__name__}" + (f" with value '{value}'." if value is not None else "."))
        super().__init__(message)
        self.key_path: str = key_path
        self.expected_type: Type = expected_type
        self.actual_type: Type = actual_type
        self.value: Any = value


class FileSystemError(PipelineError):
    """Base class for file system related errors."""
    pass


class DirectoryCreationError(FileSystemError):
    """Raised when a required directory cannot be created."""

    def __init__(self, path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to create directory {path}."
        super().__init__(message)
        self.path: str = path
        self.original_exception: Optional[Exception] = original_exception


class MissingDataFileError(FileSystemError):
    """Raised when expected raw data files are missing."""

    def __init__(self, missing_files: List[str], message: Optional[str] = None) -> None:
        if message is None:
            message = f"One or more expected raw data files are missing: {', '.join(missing_files)}."
        super().__init__(message)
        self.missing_files: List[str] = missing_files


class DownloadError(FileSystemError):
    """Raised when a file download fails."""

    def __init__(self, url: str, path: Optional[str] = None, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to download file from {url}" + (f" to {path}." if path else ".")
        super().__init__(message)
        self.url: str = url
        self.path: Optional[str] = path
        self.original_exception: Optional[Exception] = original_exception


class ExtractionError(FileSystemError):
    """Raised when a file extraction (e.g., tar.gz) fails."""

    def __init__(self, file_path: str, extract_dir: Optional[str] = None, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to extract archive {file_path}" + (f" to {extract_dir}." if extract_dir else ".")
        super().__init__(message)
        self.file_path: str = file_path
        self.extract_dir: Optional[str] = extract_dir
        self.original_exception: Optional[Exception] = original_exception


class ChunkSavingError(FileSystemError):
    """Raised when saving a processed data chunk to disk fails."""

    def __init__(self, chunk_path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to save processed data chunk to {chunk_path}."
        super().__init__(message)
        self.chunk_path: str = chunk_path
        self.original_exception: Optional[Exception] = original_exception


class DataLoadingError(PipelineError):
    """Base class for errors during the loading of raw data files (SDF, CSV, MAT)."""
    pass


class SDFParseError(DataLoadingError):
    """Raised when there's an error parsing or loading the SDF file."""

    def __init__(self, path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to load or parse SDF file: {path}."
        super().__init__(message)
        self.path: str = path
        self.original_exception: Optional[Exception] = original_exception


class CSVParseError(DataLoadingError):
    """Raised when there's an error parsing or loading the CSV file."""

    def __init__(self, path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to load or parse CSV file: {path}."
        super().__init__(message)
        self.path: str = path
        self.original_exception: Optional[Exception] = original_exception


class MatParseError(DataLoadingError):
    """Raised when there's an error parsing or loading the .mat file."""

    def __init__(self, path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to load or parse .mat file: {path}."
        super().__init__(message)
        self.path: str = path
        self.original_exception: Optional[Exception] = original_exception


class DataConsistencyError(DataLoadingError):
    """Raised when inconsistencies are found between loaded data sources (e.g., mismatched counts)."""

    def __init__(self, message: str = "Inconsistency detected between loaded data sources.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class TransformError(PipelineError):
    """Base class for errors related to PyTorch Geometric transforms."""
    pass


class TransformNotFoundError(TransformError):
    """Raised when a specified transform class is not found."""

    def __init__(self, transform_name: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"PyG transform '{transform_name}' not found in torch_geometric.transforms or custom modules."
        super().__init__(message)
        self.transform_name: str = transform_name


class TransformInstantiationError(TransformError):
    """Raised when a transform cannot be instantiated, often due to invalid kwargs."""

    def __init__(self, transform_name: str, kwargs: Dict[str, Any], message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Error instantiating PyG transform '{transform_name}' with args {kwargs}. Check kwargs in config.yaml."
        super().__init__(message)
        self.transform_name: str = transform_name
        self.kwargs: Dict[str, Any] = kwargs
        self.original_exception: Optional[Exception] = original_exception


class GraphCreationError(PipelineError):
    """Raised when an error occurs during the creation of a PyG Data object for a single molecule."""

    def __init__(self, original_idx: int, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to create PyTorch Geometric graph for molecule with original index {original_idx}."
        super().__init__(message)
        self.original_idx: int = original_idx
        self.original_exception: Optional[Exception] = original_exception


class FilterError(PipelineError):
    """Raised when an error occurs during the application of a pre-filter function."""

    def __init__(self, original_idx: int, filter_name: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Error applying filter '{filter_name}' to molecule {original_idx}."
        super().__init__(message)
        self.original_idx: int = original_idx
        self.filter_name: str = filter_name
        self.original_exception: Optional[Exception] = original_exception


class DownloadSetupError(PipelineError):
    """Raised when an error occurs during the data download and directory setup phase."""

    def __init__(self, message: str = "Error during data download/setup from db_dl.py.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class GraphProcessingError(PipelineError):
    """Raised when an error occurs during the initial graph processing phase (pyg_qm7_processing)."""

    def __init__(self, message: str = "An error occurred during initial graph processing.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class GraphCurationError(PipelineError):
    """Raised when an error occurs during the graph curation phase (qm7_curation)."""

    def __init__(self, message: str = "An error occurred during graph curation.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class CitationLoggingError(PipelineError):
    """Raised when logging citations encounters an error."""

    def __init__(self, message: str = "Failed to log citations.", original_exception: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_exception: Optional[Exception] = original_exception


class CitationFormatError(PipelineError):
    """Base class for errors specific to citation formatting."""
    pass


class InvalidCitationDataError(CitationFormatError):
    """Raised when citation data provided is not in the expected dictionary format."""

    def __init__(self, received_type: Type, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Invalid input: citation_data must be a dictionary. Received type: {received_type}."
        super().__init__(message)
        self.received_type: Type = received_type


class MalformedCitationFieldWarning(CitationFormatError):
    """Raised when a specific field in a citation has an unexpected or malformed type."""

    def __init__(self, citation_key: str, field_name: str, value: Any, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Citation key '{citation_key}' has '{field_name}' field of unexpected type or format. Value: {value}."
        super().__init__(message)
        self.citation_key: str = citation_key
        self.field_name: str = field_name
        self.value: Any = value
        self.original_exception: Optional[Exception] = original_exception


class CitationProcessingError(CitationFormatError):
    """Raised for general issues encountered during the processing of a single citation entry."""

    def __init__(self, citation_key: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"An error occurred while processing citation with key '{citation_key}'."
        super().__init__(message)
        self.citation_key: str = citation_key
        self.original_exception: Optional[Exception] = original_exception


class QM7SpecificProcessingError(PipelineError):
    """Base class for errors specifically from pyg_qm7_processing.py module."""
    pass


class QM7DataLoadError(QM7SpecificProcessingError):
    """Raised when there's an issue loading primary data files (SDF, CSV, MAT) within pyg_qm7_processing.py."""

    def __init__(self, file_path: str, data_type: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to load {data_type} data from {file_path} in pyg_qm7_processing."
        super().__init__(message)
        self.file_path: str = file_path
        self.data_type: str = data_type
        self.original_exception: Optional[Exception] = original_exception


class QM7DataAlignmentError(QM7SpecificProcessingError):
    """Raised when data loaded from different sources (SDF, MAT, energies) show critical alignment or size inconsistencies
    specific to the expectations of pyg_qm7_processing.py."""

    def __init__(self, message: str = "Data inconsistency or alignment error detected in pyg_qm7_processing across SDF, MAT, or energy files.", details: Optional[str] = None) -> None:
        super().__init__(message)
        self.details: Optional[str] = details


class QM7GraphConstructionError(GraphCreationError):
    """Raised when an error occurs during the creation of a PyG Data object for a single molecule in pyg_qm7_processing.py."""

    def __init__(self, original_idx: int, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to create PyG graph for molecule at original index {original_idx} in pyg_qm7_processing."
        super().__init__(original_idx, message, original_exception)


class QM7InvalidMoleculeError(QM7GraphConstructionError):
    """Raised when an RDKit molecule object is invalid (e.g., None, no atoms, no conformer) during pyg_qm7_processing.py's graph creation."""

    def __init__(self, original_idx: int, reason: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Invalid RDKit molecule for original index {original_idx} in pyg_qm7_processing: {reason}"
        super().__init__(original_idx, message, original_exception)
        self.reason: str = reason


class QM7FeatureExtractionError(QM7GraphConstructionError):
    """Raised when there's an error extracting features (node or edge) for a molecule within pyg_qm7_processing.py."""

    def __init__(self, original_idx: int, feature_type: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to extract {feature_type} features for molecule at original index {original_idx} in pyg_qm7_processing."
        super().__init__(original_idx, message, original_exception)
        self.feature_type: str = feature_type


class QM7EdgeFeatureMismatchError(QM7GraphConstructionError):
    """Raised when there's a size or indexing mismatch between RDKit bond features and Coulomb Matrix edge features for a molecule."""

    def __init__(self, original_idx: int, rdkit_edge_count: int, cm_edge_count: int, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Edge feature count mismatch for molecule {original_idx} in pyg_qm7_processing: RDKit ({rdkit_edge_count}) vs CM ({cm_edge_count})."
        super().__init__(original_idx, message, original_exception)
        self.rdkit_edge_count: int = rdkit_edge_count
        self.cm_edge_count: int = cm_edge_count


class QM7ChunkSavingError(QM7SpecificProcessingError):
    """Raised when there's an error saving a chunk of processed PyG Data objects within pyg_qm7_processing.py."""

    def __init__(self, chunk_path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to save processed data chunk to {chunk_path} in pyg_qm7_processing."
        super().__init__(message)
        self.chunk_path: str = chunk_path
        self.original_exception: Optional[Exception] = original_exception


class QM7PreFilterError(QM7SpecificProcessingError):
    """Raised when a pre_filter function encounters an error during execution within pyg_qm7_processing.py."""

    def __init__(self, original_idx: int, filter_name: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Error applying pre-filter '{filter_name}' to molecule {original_idx} in pyg_qm7_processing."
        super().__init__(message)
        self.original_idx: int = original_idx
        self.filter_name: str = filter_name
        self.original_exception: Optional[Exception] = original_exception


class QM7CurationError(PipelineError):
    """Base class for exceptions specific to qm7_curation.py module."""
    pass


class ChunkFileAccessError(QM7CurationError, FileSystemError):
    """Raised when there's an issue accessing or finding chunk files in the specified directory."""

    def __init__(self, chunk_dir: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to access or find chunk files in directory: {chunk_dir}"
        super().__init__(message)
        self.chunk_dir: str = chunk_dir
        self.original_exception: Optional[Exception] = original_exception


class ChunkFileLoadError(QM7CurationError, DataLoadingError):
    """Raised when a chunk file fails to load correctly (e.g., corrupted file, not a PyTorch object)."""

    def __init__(self, file_path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to load chunk file: {file_path}. It might be corrupted or not a valid PyTorch object."
        super().__init__(message)
        self.file_path: str = file_path
        self.original_exception: Optional[Exception] = original_exception


class InvalidDataInChunkError(QM7CurationError, DataConsistencyError):
    """Raised when an item within a loaded chunk is not a torch_geometric.data.Data object."""

    def __init__(self, chunk_file: str, item_index: int, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Item {item_index} in chunk file {chunk_file} is not a PyTorch Geometric Data object."
        super().__init__(message)
        self.chunk_file: str = chunk_file
        self.item_index: int = item_index


class FeatureConversionError(QM7CurationError, TransformError):
    """Raised when a feature tensor cannot be converted to a NumPy array of type float for StandardScaler processing."""

    def __init__(self, chunk_file: str, item_index: int, feature_key: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Could not convert feature '{feature_key}' for item {item_index} in chunk {chunk_file} from tensor to numpy float."
        super().__init__(message)
        self.chunk_file: str = chunk_file
        self.item_index: int = item_index
        self.feature_key: str = feature_key
        self.original_exception: Optional[Exception] = original_exception


class FeatureNormalizationError(QM7CurationError, TransformError):
    """Raised when an error occurs during the StandardScaler.transform or partial_fit operation."""

    def __init__(self, feature_key: str, chunk_file: Optional[str] = None, item_index: Optional[int] = None, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            location_info = ""
            if chunk_file and item_index is not None:
                location_info = f" for graph in chunk {chunk_file}, index {item_index}"
            message = f"Error normalizing feature '{feature_key}'{location_info}."
        super().__init__(message)
        self.feature_key: str = feature_key
        self.chunk_file: Optional[str] = chunk_file
        self.item_index: Optional[int] = item_index
        self.original_exception: Optional[Exception] = original_exception


class PreTransformApplicationError(QM7CurationError, TransformError):
    """Raised when a pre_transforms operation fails on a Data object."""

    def __init__(self, chunk_file: str, item_index: int, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Error applying pre-transforms to graph in chunk {chunk_file}, index {item_index}."
        super().__init__(message)
        self.chunk_file: str = chunk_file
        self.item_index: int = item_index
        self.original_exception: Optional[Exception] = original_exception


class DatasetSaveError(QM7CurationError, FileSystemError):
    """Raised when the final curated dataset cannot be saved to the specified output path."""

    def __init__(self, output_path: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to save curated dataset to: {output_path}."
        super().__init__(message)
        self.output_path: str = output_path
        self.original_exception: Optional[Exception] = original_exception


class ChunkDirectoryCleanupError(QM7CurationError, FileSystemError):
    """Raised if there's an issue cleaning up the intermediate chunk directory."""

    def __init__(self, chunk_dir: str, message: Optional[str] = None, original_exception: Optional[Exception] = None) -> None:
        if message is None:
            message = f"Failed to clean up intermediate chunk directory: {chunk_dir}."
        super().__init__(message)
        self.chunk_dir: str = chunk_dir
        self.original_exception: Optional[Exception] = original_exception


class MissingScalerAttributeError(QM7CurationError, TransformError):
    """Raised when a StandardScaler object is missing expected attributes (mean_, scale_) after partial_fit."""

    def __init__(self, feature_key: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"StandardScaler for feature '{feature_key}' is not fitted or missing 'mean_' or 'scale_' attributes."
        super().__init__(message)
        self.feature_key: str = feature_key
