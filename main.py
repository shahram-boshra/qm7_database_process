# main.py
import logging
from pathlib import Path
import sys
from functools import partial
from typing import Dict, Any, List, Optional, Callable, Union


try:
    from exceptions import (
        ImportConfigError, ImportModuleError, ImportPyGError,
        ConfigNotFound, ConfigAttributeMissingError, InvalidConfigError,
        MissingDownloadURLError,
        DirectoryCreationError, MissingDataFileError,
        TransformNotFoundError, TransformInstantiationError,
        DownloadSetupError, GraphProcessingError, GraphCurationError,
        CitationLoggingError,
        PipelineError
    )
except ImportError as e:
    logging.critical(f"FATAL ERROR: Could not import custom exceptions from exceptions.py: {e}.")
    logging.critical("Please ensure exceptions.py exists and is not corrupted. Exiting.")
    sys.exit(1)


current_dir: Path = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from config import Config
except ImportError as e:
    raise ImportConfigError(original_exception=e) from e

try:
    import db_dl
except ImportError as e:
    raise ImportModuleError("db_dl", original_exception=e) from e
try:
    import pyg_qm7_processing
except ImportError as e:
    raise ImportModuleError("pyg_qm7_processing", original_exception=e) from e
try:
    import qm7_curation
except ImportError as e:
    raise ImportModuleError("qm7_curation", original_exception=e) from e
try:
    import citations
except ImportError as e:
    raise ImportModuleError("citations", original_exception=e) from e

try:
    from torch_geometric.transforms import Compose
    import torch_geometric.transforms as T
    from torch_geometric.transforms import BaseTransform
    from rdkit import RDLogger
except ImportError as e:
    raise ImportPyGError(original_exception=e) from e


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
main_logger: logging.Logger = logging.getLogger(__name__)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
main_logger.info("RDKit logger level set to CRITICAL via main.py configuration to suppress verbose RDKit output.")

CUSTOM_TRANSFORM_MODULES: Dict[str, Any] = {
    "CustomEdgeFeatureCombiner": qm7_curation,
}

QM7_CITATIONS_DATA: List[Dict[str, str]] = [
    {
        "key": "Rupp2012",
        "authors": "Rupp, Matthias, et al.",
        "title": "Fast and accurate modeling of molecular atomization energies with machine learning.",
        "journal": "Physical Review Letters",
        "volume": "108",
        "issue": "5",
        "pages": "058301",
        "year": "2012",
        "doi": "10.1103/PhysRevLett.108.058301",
        "full_citation": "[1] Rupp, Matthias, et al. \"Fast and accurate modeling of molecular\n    atomization energies with machine learning.\" Physical review letters\n    108.5 (2012): 058301.\n    DOI: 10.1103/PhysRevLett.108.058301"
    },
    {
        "key": "Montavon2012",
        "authors": "Montavon, Grégoire, et al.",
        "title": "Learning invariant representations of molecules for atomization energy prediction.",
        "conference": "Advances in Neural Information Processing Systems",
        "year": "2012",
        "note": "(NIPS 2012)",
        "full_citation": "[2] Montavon, Grégoire, et al. \"Learning invariant representations of\n    molecules for atomization energy prediction.\" Advances in Neural\n    Information Processing Systems. 2012.\n    (NIPS 2012)"
    }
]


def dynamically_load_transforms(transforms_config: Optional[List[Dict[str, Any]]]) -> Compose:
    """
    Dynamically loads and instantiates PyTorch Geometric transforms from a list
    of transform configurations (name and kwargs).
    This function now supports both standard PyTorch Geometric transforms
    and custom transforms defined in other modules.

    Args:
        transforms_config (Optional[List[Dict[str, Any]]]): A list of dictionaries,
            where each dictionary specifies a transform's 'name' and its 'kwargs'
            (keyword arguments) for instantiation.

    Returns:
        Compose: A `torch_geometric.transforms.Compose` object containing all
            successfully loaded and instantiated transforms. Returns an empty
            Compose object if no transforms are specified or successfully loaded.

    Raises:
        TransformNotFoundError: If a specified transform cannot be found in either
            standard PyTorch Geometric transforms or the custom transform modules.
        TransformInstantiationError: If an error occurs during the instantiation
            of a transform, typically due to incorrect keyword arguments.
    """
    transforms: List[BaseTransform] = []
    if not transforms_config:
        main_logger.info("No transforms specified in configuration. Returning an empty Compose.")
        return Compose([])

    for transform_info in transforms_config:
        name: Optional[str] = transform_info.get('name')
        kwargs: Dict[str, Any] = transform_info.get('kwargs', {})
        if not name:
            main_logger.warning(f"Skipping transform with missing 'name' in config: {transform_info}. Each transform must have a 'name'.")
            continue
        try:
            transform_class: Optional[Any] = None
            if hasattr(T, name):
                transform_class = getattr(T, name)
            elif name in CUSTOM_TRANSFORM_MODULES:
                module = CUSTOM_TRANSFORM_MODULES[name]
                if hasattr(module, name):
                    transform_class = getattr(module, name)
                else:
                    raise TransformNotFoundError(name, message=f"Custom transform '{name}' not found in its specified module '{module.__name__}'.")
            else:
                raise TransformNotFoundError(name)

            try:
                transforms.append(transform_class(**kwargs))
                main_logger.info(f"Dynamically loaded PyG transform: {name} with args: {kwargs}")
            except TypeError as e:
                raise TransformInstantiationError(name, kwargs, original_exception=e) from e
            except Exception as e:
                main_logger.error(f"An unexpected error occurred while instantiating transform '{name}': {e}. Skipping this transform.")

        except TransformNotFoundError as e:
            main_logger.error(f"{e} Skipping this transform.")
            continue
        except TransformInstantiationError as e:
            main_logger.error(f"{e} Skipping this transform.")
            continue
        except Exception as e:
            main_logger.error(f"An unexpected error occurred while attempting to locate or prepare transform '{name}': {e}. Skipping this transform.")
            continue

    return Compose(transforms) if transforms else Compose([])


def main() -> None:
    """
    Orchestrates the complete QM7 data pipeline, now using a centralized
    configuration loaded and passed directly to functions.

    This function performs the following steps:
    1. Loads configuration from `config.yaml`.
    2. Downloads raw QM7 dataset files (SDF and MAT) if not already present.
    3. Processes raw data into PyTorch Geometric graph chunks, applying
        pre-filters as defined in the configuration.
    4. Curates the processed graphs, including normalization and applying
        additional PyTorch Geometric transforms.
    5. Logs relevant citations for the QM7 dataset.

    Raises:
        ConfigNotFound: If `config.yaml` is not found.
        InvalidConfigError: If the configuration file is found but cannot be parsed.
        ConfigAttributeMissingError: If a required attribute is missing from the configuration.
        MissingDownloadURLError: If download URLs are missing from the configuration during data setup.
        DownloadSetupError: If an error occurs during the data download or directory setup phase.
        MissingDataFileError: If essential raw data files are not found after the download step.
        DirectoryCreationError: If there's an issue creating necessary directories (e.g., intermediate chunks, processed data).
        GraphProcessingError: If an error occurs during the initial processing of raw data into graphs.
        GraphCurationError: If an error occurs during the curation of graphs (normalization, applying transforms).
        CitationLoggingError: If an error occurs while logging citations.
        PipelineError: A general catch-all for unexpected pipeline failures.
    """
    main_logger.info("--- Starting Orchestrated QM7 Data Pipeline ---")

    config_path: Path = current_dir / "config.yaml"
    cfg: Optional[Config] = None
    try:
        if not config_path.exists():
            raise ConfigNotFound(config_path)
        cfg = Config(config_path)
        main_logger.info("Configuration loaded successfully from config.yaml.")
    except ConfigNotFound as e:
        main_logger.critical(f"{e} Exiting.")
        sys.exit(1)
    except Exception as e:
        raise InvalidConfigError(config_path, original_exception=e) from e

    try:
        base_data_dir: Path = Path(cfg.DATA_PATHS.BASE_DATA_DIR)
        qm7_sdf_url: str = cfg.DOWNLOAD_URLS.GDB7_SDF_URL
        qm7_mat_url: str = cfg.DOWNLOAD_URLS.QM7_MAT_URL

        chunk_size: int = cfg.PROCESSING.CHUNK_SIZE

        curated_output_file_name: str = cfg.OUTPUT_FILES.CURATED_DATASET_NAME
        feature_keys_for_norm: List[str] = cfg.CURATION.get('FEATURE_KEYS_FOR_NORMALIZATION', [])
        additional_pyg_transforms_config: List[Dict[str, Any]] = cfg.CURATION.get('ADDITIONAL_PYG_TRANSFORMS', [])

        pre_filters_config: Dict[str, Any] = cfg.PROCESSING.get('PRE_FILTERS', {})

    except AttributeError as e:
        attribute_path: str = str(e).split("'")[1] if "'" in str(e) else "unknown attribute"
        raise ConfigAttributeMissingError(attribute_path, original_exception=e) from e
    except Exception as e:
        main_logger.critical(f"An unexpected error occurred while accessing configuration attributes: {e}. Exiting.")
        sys.exit(1)

    main_logger.info(f"Base data directory from config: {base_data_dir}")
    main_logger.info(f"Processing chunk size from config: {chunk_size}")

    raw_dir: Path = base_data_dir / "qm7" / "raw"
    processed_dir: Path = base_data_dir / "qm7" / "processed"
    intermediate_chunk_dir: Path = processed_dir / "qm7_intermediate_chunks"
    curated_output_file: Path = processed_dir / curated_output_file_name

    sdf_file: Path = raw_dir / "gdb7.sdf"
    energies_file: Path = raw_dir / "gdb7.sdf.csv"
    mat_file: Path = raw_dir / "qm7.mat"

    main_logger.info("\n--- Step 1: Running data download and directory setup ---")
    try:
        db_dl.main(
            base_data_dir=base_data_dir,
            qm7_sdf_url=qm7_sdf_url,
            qm7_mat_url=qm7_mat_url
        )
        main_logger.info("db_dl.py main function completed. Raw data should be in place.")
    except MissingDownloadURLError as e:
        main_logger.critical(f"{e} Aborting pipeline as raw data cannot be obtained.")
        return
    except Exception as e:
        raise DownloadSetupError(message=f"Error during data download/setup from db_dl.py: {e}", original_exception=e) from e

    missing_files: List[str] = []
    if not sdf_file.exists():
        missing_files.append(str(sdf_file))
    if not energies_file.exists():
        missing_files.append(str(energies_file))
    if not mat_file.exists():
        missing_files.append(str(mat_file))

    if missing_files:
        raise MissingDataFileError(missing_files)


    active_pre_filters: List[Callable[..., Any]] = []
    if pre_filters_config.get('enable_max_nodes_filter', False):
        try:
            max_nodes_val: Optional[int] = pre_filters_config.get('max_nodes')
            if max_nodes_val is None:
                main_logger.warning("Configuration warning: 'max_nodes' for 'enable_max_nodes_filter' is missing. Skipping this filter.")
            else:
                active_pre_filters.append(partial(pyg_qm7_processing.filter_by_max_nodes, max_nodes=max_nodes_val))
                main_logger.info(f"Enabled pre-filter: max_nodes={max_nodes_val}")
        except Exception as e:
            main_logger.error(f"Error configuring 'max_nodes' filter: {e}. Skipping this filter.")

    if pre_filters_config.get('enable_min_nodes_filter', False):
        try:
            min_nodes_val: Optional[int] = pre_filters_config.get('min_nodes')
            if min_nodes_val is None:
                main_logger.warning("Configuration warning: 'min_nodes' for 'enable_min_nodes_filter' is missing. Skipping this filter.")
            else:
                active_pre_filters.append(partial(pyg_qm7_processing.filter_by_min_nodes, min_nodes=min_nodes_val))
                main_logger.info(f"Enabled pre-filter: min_nodes={min_nodes_val}")
        except Exception as e:
            main_logger.error(f"Error configuring 'min_nodes' filter: {e}. Skipping this filter.")

    if pre_filters_config.get('enable_num_carbons_filter', False):
        try:
            min_carbons_val: Optional[int] = pre_filters_config.get('min_carbons')
            if min_carbons_val is None:
                main_logger.warning("Configuration warning: 'min_carbons' for 'enable_num_carbons_filter' is missing. Skipping this filter.")
            else:
                active_pre_filters.append(partial(pyg_qm7_processing.filter_by_num_carbons, min_carbons=min_carbons_val))
                main_logger.info(f"Enabled pre-filter: min_carbons={min_carbons_val}")
        except Exception as e:
            main_logger.error(f"Error configuring 'num_carbons' filter: {e}. Skipping this filter.")

    pre_filter_composed: Optional[Compose] = Compose(active_pre_filters) if active_pre_filters else None


    main_logger.info("\n--- Step 2: Processing raw data into PyTorch Geometric graphs (with filtering) ---")
    try:
        try:
            intermediate_chunk_dir.mkdir(parents=True, exist_ok=True)
            main_logger.info(f"Ensured intermediate chunk directory exists: {intermediate_chunk_dir}")
        except OSError as e:
            raise DirectoryCreationError(intermediate_chunk_dir, original_exception=e) from e

        returned_chunk_dir: Path = pyg_qm7_processing.process_qm7_data(
            sdf_file=sdf_file,
            energies_file=energies_file,
            mat_file=mat_file,
            intermediate_chunk_output_dir=intermediate_chunk_dir,
            chunk_size=chunk_size,
            pre_filter=pre_filter_composed
        )
        if returned_chunk_dir and returned_chunk_dir.exists():
            main_logger.info(f"Initial PyTorch Geometric graphs processed and saved to intermediate chunk directory: {returned_chunk_dir}")
            intermediate_chunk_dir = returned_chunk_dir
        else:
            raise GraphProcessingError(message="pyg_qm7_processing.process_qm7_data failed to return a valid, existing chunk directory.")

    except (DirectoryCreationError, GraphProcessingError) as e:
        main_logger.error(f"{e} Aborting pipeline.")
        main_logger.error("--- QM7 Data Pipeline FAILED ---")
        return
    except Exception as e:
        raise GraphProcessingError(message=f"An unexpected error occurred during initial graph processing: {e}", original_exception=e) from e


    main_logger.info("\n--- Step 3: Curating the graphs (normalization and pre_transforms) ---")
    try:
        additional_pyg_transforms: Compose = dynamically_load_transforms(
            additional_pyg_transforms_config
        )

        try:
            processed_dir.mkdir(parents=True, exist_ok=True)
            main_logger.info(f"Ensured processed data directory exists: {processed_dir}")
        except OSError as e:
            raise DirectoryCreationError(processed_dir, original_exception=e) from e

        qm7_curation.curate_qm7_data(
            chunk_dir=intermediate_chunk_dir,
            output_path=curated_output_file,
            pre_transforms=additional_pyg_transforms,
            feature_keys_for_norm=feature_keys_for_norm
        )
        if curated_output_file.exists():
            main_logger.info(f"PyTorch Geometric dataset successfully curated and saved to: {curated_output_file}")
            main_logger.info("--- QM7 Data Pipeline Completed Successfully ---")
        else:
            main_logger.error(f"Curation completed, but the expected output file was not found: {curated_output_file}. There might have been an issue during saving.")
            main_logger.info("--- QM7 Data Pipeline FAILED (Curation Output Missing) ---")

    except (DirectoryCreationError, GraphCurationError) as e:
        main_logger.error(f"{e} Aborting pipeline.")
        main_logger.error("--- QM7 Data Pipeline FAILED ---")
        return
    except Exception as e:
        raise GraphCurationError(message=f"An unexpected error occurred during graph curation: {e}", original_exception=e) from e

    main_logger.info("\nReminder: To verify the generated dataset, consider running 'chk_qm7_dataset.py' on the output file.")
    main_logger.info("For custom filtering/transforms, modify the 'pre_filters' and 'additional_pyg_transforms' sections in 'config.yaml'.")

    try:
        citations.log_qm7_citations(QM7_CITATIONS_DATA)
    except Exception as e:
        raise CitationLoggingError(original_exception=e) from e


if __name__ == "__main__":
    try:
        main()
    except (
        ImportConfigError, ImportModuleError, ImportPyGError,
        ConfigNotFound, ConfigAttributeMissingError, InvalidConfigError,
        MissingDownloadURLError,
        MissingDataFileError,
        DownloadSetupError, GraphProcessingError, GraphCurationError,
        CitationLoggingError,
        PipelineError
    ) as e:
        main_logger.critical(f"A critical pipeline error occurred: {e}")
        if hasattr(e, 'original_exception') and e.original_exception:
            main_logger.critical(f"Original exception details: {e.original_exception}")
        sys.exit(1)
    except Exception as e:
        main_logger.critical(f"An unexpected and unhandled error caused the pipeline to fail: {e}", exc_info=True)
        sys.exit(1)
