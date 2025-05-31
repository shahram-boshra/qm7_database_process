# qm7_curation.py

import logging
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import shutil
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose, BaseTransform
from typing import Callable, Union, List, Optional, Dict
import torch_geometric.data.data
import torch_geometric.data.storage

from exceptions import (
    PipelineError,
    GraphCurationError,
    ChunkFileAccessError,
    ChunkFileLoadError,
    InvalidDataInChunkError,
    FeatureConversionError,
    FeatureNormalizationError,
    PreTransformApplicationError,
    DatasetSaveError,
    ChunkDirectoryCleanupError,
    MissingScalerAttributeError
)

logger = logging.getLogger(__name__)

try:
    safe_classes = [
        torch_geometric.data.data.Data,
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
        Compose,
    ]
    torch.serialization.add_safe_globals(safe_classes)
    logger.debug("Successfully added PyTorch Geometric Data classes to torch.serialization.add_safe_globals.")
except AttributeError:
    logger.warning("torch.serialization.add_safe_globals not available. "
                                "If using PyTorch 2.6+, ensure compatibility or manually handle weights_only.")
except Exception as e:
    logger.error(f"Unexpected error when adding safe globals: {e}")


def calculate_global_stats_with_scaler(chunk_files: List[Path], feature_keys: List[str]) -> Dict[str, StandardScaler]:
    """
    Calculates global mean and standard deviation for specified features across all chunks
    using scikit-learn's StandardScaler with partial_fit for memory efficiency.

    This function iterates through chunk files, loads PyTorch Geometric `Data` objects,
    and applies `partial_fit` for each specified feature to a dedicated `StandardScaler`.
    It handles various error conditions, including missing chunk files, invalid data types
    within chunks, issues during tensor-to-numpy conversion, and constant features.

    Args:
        chunk_files (List[Path]): A list of file paths to the PyTorch Geometric Data chunks.
        feature_keys (List[str]): A list of string keys corresponding to the attributes
                                  in the `Data` object (e.g., 'x' for node features,
                                  'edge_attr' for edge features) for which global statistics
                                  (mean and standard deviation) should be calculated.

    Returns:
        Dict[str, StandardScaler]: A dictionary where keys are feature names and values are
                                   fitted `StandardScaler` objects containing the calculated
                                   global mean and standard deviation for that feature.

    Raises:
        ChunkFileAccessError: If no chunk files are provided or if there's an issue
                              accessing the chunk directory.
        ChunkFileLoadError: If a chunk file fails to load or is unreadable.
        InvalidDataInChunkError: If a loaded chunk is not a list or contains items
                                 that are not `PyG Data` objects.
        FeatureConversionError: If a feature tensor cannot be converted to a NumPy array
                                of float type for `StandardScaler`.
        FeatureNormalizationError: If an unexpected error occurs during `partial_fit`
                                   of the `StandardScaler`.
    """
    logger.info(f"Calculating global statistics for features: {feature_keys} using StandardScaler (Pass 1/2)")

    scalers: Dict[str, StandardScaler] = {key: StandardScaler() for key in feature_keys}

    if not chunk_files:
        raise ChunkFileAccessError(
            chunk_dir=chunk_files[0].parent if chunk_files else Path("N/A"),
            message=f"No chunk files provided for global statistics calculation."
        )

    for chunk_file in tqdm(chunk_files, desc="Pass 1/2: Accumulating Stats with StandardScaler"):
        try:
            chunk: List[Data] = torch.load(chunk_file)
            if not isinstance(chunk, list):
                logger.warning(f"Chunk file {chunk_file} did not load as a list of Data objects. Skipping.")
                raise InvalidDataInChunkError(
                    chunk_file=chunk_file,
                    item_index=-1,
                    message=f"Chunk file {chunk_file} loaded but is not a list."
                )
        except InvalidDataInChunkError:
            continue
        except Exception as e:
            logger.error(f"Error loading chunk file {chunk_file}: {e}. Skipping this chunk.")
            raise ChunkFileLoadError(
                file_path=chunk_file,
                message=f"Failed to load or parse chunk file {chunk_file}",
                original_exception=e
            )

        for i, data in enumerate(chunk):
            if not isinstance(data, Data):
                logger.warning(f"Item {i} in chunk {chunk_file} is not a PyG Data object. Skipping.")
                continue

            for key in feature_keys:
                try:
                    value: Optional[torch.Tensor] = getattr(data, key, None)
                    if value is not None and value.numel() > 0:
                        try:
                            value_np: np.ndarray = value.detach().cpu().numpy().astype(float)
                        except Exception as e:
                            logger.warning(f"Could not convert '{key}' from tensor to numpy float for graph in chunk {chunk_file}, index {i}: {e}. Skipping normalization for this attribute.")
                            raise FeatureConversionError(
                                chunk_file=chunk_file,
                                item_index=i,
                                feature_key=key,
                                message=f"Failed to convert feature '{key}' to numpy float.",
                                original_exception=e
                            )

                        if value_np.ndim == 1:
                            value_np = value_np.reshape(-1, 1)
                        elif value_np.ndim == 0:
                            value_np = np.array([[value_np.item()]])

                        if value_np.size == 0:
                            logger.debug(f"Skipping empty feature '{key}' in chunk {chunk_file}, index {i} during partial_fit.")
                            continue

                        if value_np.shape[0] > 0 and np.std(value_np) < 1e-9:
                            logger.warning(f"Feature '{key}' in chunk {chunk_file}, index {i} has effectively zero standard deviation. Skipping normalization for this attribute as it's constant.")
                            continue

                        try:
                            scalers[key].partial_fit(value_np)
                        except Exception as e:
                            logger.error(f"Unexpected error during partial_fit for '{key}' in chunk {chunk_file}, index {i}: {e}")
                            raise FeatureNormalizationError(
                                feature_key=key,
                                chunk_file=chunk_file,
                                item_index=i,
                                message=f"Unexpected error during partial_fit for '{key}'.",
                                original_exception=e
                            )
                    else:
                        logger.debug(f"Feature '{key}' in chunk {chunk_file}, index {i} is None or empty. Skipping for stats calculation.")
                except (AttributeError, FeatureConversionError, FeatureNormalizationError):
                    continue
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing feature '{key}' for graph in chunk {chunk_file}, index {i}: {e}")
                    continue

    logger.info("Global statistics calculation complete.")
    for key, scaler in scalers.items():
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and scaler.scale_.size > 0 and not np.any(scaler.scale_ == 0):
            mean_val: Union[List[float], float] = scaler.mean_.tolist() if isinstance(scaler.mean_, np.ndarray) else scaler.mean_
            std_val: Union[List[float], float] = scaler.scale_.tolist() if isinstance(scaler.scale_, np.ndarray) else scaler.scale_
        else:
            logger.warning(f"  Feature '{key}': StandardScaler not fitted (no data or constant/zero-variance values) or attributes missing. Normalization for this feature will be skipped during the second pass.")
    return scalers


class CustomEdgeFeatureCombiner(BaseTransform):
    """
    A custom PyTorch Geometric transform to combine or modify edge features.
    This is a placeholder; you'll need to implement the actual logic
    for combining features based on your specific requirements.

    This transform is designed to be part of a `torch_geometric.transforms.Compose`
    pipeline. It operates on the `edge_attr` attribute of a `Data` object.

    Attributes:
        kwargs (dict): Keyword arguments passed during initialization, intended
                       for configuring the combination logic (currently not used
                       in the placeholder implementation).
    """
    def __init__(self, **kwargs: dict) -> None:
        """
        Initializes the CustomEdgeFeatureCombiner.

        Args:
            **kwargs (dict): Arbitrary keyword arguments.
        """
        super().__init__()
        self.kwargs = kwargs
        logger.info(f"CustomEdgeFeatureCombiner initialized with kwargs: {kwargs}")

    def __call__(self, data: Data) -> Data:
        """
        Applies the edge feature combination logic to the input `Data` object.

        Args:
            data (Data): The PyTorch Geometric `Data` object.

        Returns:
            Data: The modified PyTorch Geometric `Data` object with combined/modified
                  edge features.

        Raises:
            PreTransformApplicationError: If an error occurs during the application
                                          of the custom edge feature combination logic.
        """
        try:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0:
                logger.debug(f"Applying CustomEdgeFeatureCombiner: Original edge_attr shape {data.edge_attr.shape}")
            else:
                logger.debug("CustomEdgeFeatureCombiner called, but data has no 'edge_attr' or it is None/empty. No combination applied.")
        except Exception as e:
            logger.error(f"Error applying CustomEdgeFeatureCombiner to data object: {e}")
            raise PreTransformApplicationError(
                chunk_file="N/A",
                item_index="N/A",
                message=f"Error within CustomEdgeFeatureCombiner: {e}",
                original_exception=e
            )
        return data

    def __repr__(self) -> str:
        """
        Returns a string representation of the transform.
        """
        return f"{self.__class__.__name__}(kwargs={self.kwargs})"


class StandardScalerTransform(BaseTransform):
    """
    A PyTorch Geometric transform to apply StandardScaler normalization to a specified feature.

    This transform takes a pre-fitted `sklearn.preprocessing.StandardScaler` object
    and applies its `transform` method to a designated feature (e.g., 'x', 'edge_attr')
    within a PyTorch Geometric `Data` object. It handles reshaping for 0D or 1D tensors
    to be compatible with `StandardScaler` and converts the result back to a PyTorch tensor.

    Attributes:
        feature_key (str): The name of the feature attribute in the `Data` object
                           to be normalized (e.g., 'x', 'edge_attr').
        scaler (StandardScaler): The pre-fitted `StandardScaler` instance to use
                                 for normalization.
        _is_fitted (bool): Internal flag indicating whether the provided scaler
                           is properly fitted and can perform transformations.
    """
    def __init__(self, feature_key: str, scaler: StandardScaler):
        """
        Initializes the StandardScalerTransform.

        Args:
            feature_key (str): The name of the feature attribute in the `Data` object
                               to be normalized.
            scaler (StandardScaler): A pre-fitted `StandardScaler` instance.
        """
        super().__init__()
        self.feature_key = feature_key
        self.scaler = scaler
        if not (hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and scaler.scale_.size > 0 and not np.any(scaler.scale_ == 0)):
            logger.warning(f"StandardScaler for '{feature_key}' is not fully fitted (missing mean/scale or zero scale). This transform will skip normalization for this feature.")
            self._is_fitted = False
        else:
            self._is_fitted = True

    def __call__(self, data: Data) -> Data:
        """
        Applies StandardScaler normalization to the specified feature of the input `Data` object.

        Args:
            data (Data): The PyTorch Geometric `Data` object.

        Returns:
            Data: The modified PyTorch Geometric `Data` object with the specified feature normalized.

        Raises:
            FeatureNormalizationError: If an error occurs during the normalization process,
                                       such as issues with tensor-to-numpy conversion,
                                       `scaler.transform` application, or conversion back
                                       to PyTorch tensor.
        """
        if not self._is_fitted:
            logger.debug(f"Skipping StandardScalerTransform for '{self.feature_key}' as its scaler is not fitted.")
            return data

        if hasattr(data, self.feature_key) and getattr(data, self.feature_key) is not None:
            original_tensor = getattr(data, self.feature_key)
            if original_tensor.numel() == 0:
                logger.debug(f"Skipping normalization for empty feature '{self.feature_key}' in data object.")
                return data

            original_dtype = original_tensor.dtype
            original_device = original_tensor.device

            try:
                value_np = original_tensor.detach().cpu().numpy().astype(float)
                original_ndim = value_np.ndim

                if value_np.ndim == 0:
                    value_np = np.array([[value_np.item()]])
                elif value_np.ndim == 1:
                    value_np = value_np.reshape(-1, 1)

                transformed_value_np = self.scaler.transform(value_np)
                new_tensor = torch.tensor(transformed_value_np, dtype=original_dtype, device=original_device)

                if original_ndim == 0:
                    new_tensor = new_tensor.squeeze()
                elif original_ndim == 1:
                    new_tensor = new_tensor.squeeze(dim=-1)

                setattr(data, self.feature_key, new_tensor)

            except Exception as e:
                logger.error(f"Error during StandardScalerTransform for feature '{self.feature_key}': {e}")
                raise FeatureNormalizationError(
                    feature_key=self.feature_key,
                    message=f"Error applying StandardScalerTransform for '{self.feature_key}'",
                    original_exception=e
                ) from e
        else:
            logger.debug(f"Data object has no attribute '{self.feature_key}' or it is None/empty. Skipping StandardScalerTransform.")
        return data

    def __repr__(self) -> str:
        """
        Returns a string representation of the transform, including its feature key and fit status.
        """
        return f"{self.__class__.__name__}(feature_key='{self.feature_key}', fitted={self._is_fitted})"


def curate_qm7_data(
    chunk_dir: Path,
    output_path: Path,
    feature_keys_for_norm: List[str],
    pre_transforms: Optional[Compose] = None
) -> None:
    """
    Orchestrates the curation of the QM7 dataset, including normalization and additional transformations.

    This function performs a two-pass process:
    1. **Pass 1 (Statistics Calculation):** It iterates through the chunked data
       to calculate global mean and standard deviation for specified features
       using `sklearn.preprocessing.StandardScaler.partial_fit`.
    2. **Pass 2 (Normalization and Transformation):** It then iterates through the
       data again, applying the fitted `StandardScaler` transformations
       and any additional user-defined `pre_transforms` (from `torch_geometric.transforms.Compose`).
    Finally, the curated data is saved to a single file, and the intermediate
    chunk directory is cleaned up.

    Args:
        chunk_dir (Path): The path to the directory containing the intermediate
                          chunked PyTorch Geometric Data files.
        output_path (Path): The desired path where the final, curated dataset
                            (a list of `Data` objects) will be saved as a single
                            PyTorch .pt file.
        feature_keys_for_norm (List[str]): A list of string keys (e.g., 'x', 'edge_attr')
                                           corresponding to the attributes in the `Data`
                                           object that should be normalized using
                                           `StandardScaler`.
        pre_transforms (Optional[Compose]): An optional `torch_geometric.transforms.Compose`
                                            object containing additional transformations
                                            to apply to each graph *after* normalization.

    Raises:
        GraphCurationError: A general error indicating a failure in the curation pipeline,
                            wrapping more specific exceptions.
        ChunkFileAccessError: If there's an issue accessing the chunk files.
        ChunkFileLoadError: If a chunk file cannot be loaded during either pass.
        InvalidDataInChunkError: If data within a chunk is not in the expected format.
        FeatureConversionError: If a feature cannot be converted for normalization.
        FeatureNormalizationError: If an error occurs during the normalization process.
        PreTransformApplicationError: If an error occurs while applying a custom
                                      `pre_transform`.
        DatasetSaveError: If the final curated dataset fails to save.
        ChunkDirectoryCleanupError: If there's an issue cleaning up the intermediate
                                    chunk directory.
    """
    logger.info("--- Starting QM7 Data Curation ---")

    chunk_files: List[Path] = []
    try:
        chunk_files = sorted([f for f in chunk_dir.glob('chunk_*.pt') if f.is_file()])
        if not chunk_files:
            logger.warning(f"No chunk files found in {chunk_dir}. No data to curate.")
            return
    except Exception as e:
        logger.error(f"Error accessing chunk directory or files in {chunk_dir}: {e}. Curation aborted.")
        raise ChunkFileAccessError(
            chunk_dir=chunk_dir,
            message=f"Error accessing chunk files during initial scan: {e}",
            original_exception=e
        )

    global_scalers: Dict[str, StandardScaler] = {}
    try:
        global_scalers = calculate_global_stats_with_scaler(chunk_files, feature_keys=feature_keys_for_norm)
    except (ChunkFileAccessError, ChunkFileLoadError, FeatureConversionError, FeatureNormalizationError) as e:
        logger.error(f"Error during global statistics calculation: {e}. Curation aborted.")
        raise GraphCurationError(message=f"Curation aborted due to statistics calculation error: {e}", original_exception=e)
    except Exception as e:
        logger.error(f"An unexpected error occurred during global statistics calculation: {e}. Curation aborted.")
        raise GraphCurationError(message=f"Curation aborted due to unexpected error during stats calculation: {e}", original_exception=e)


    normalization_transforms: List[BaseTransform] = []
    for key in feature_keys_for_norm:
        scaler = global_scalers.get(key)
        if scaler and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_') and scaler.scale_.size > 0 and not np.any(scaler.scale_ == 0):
            normalization_transforms.append(StandardScalerTransform(key, scaler))
        else:
            logger.warning(f"Normalization for feature '{key}' will be skipped during Pass 2 as its StandardScaler was not properly fitted.")

    all_transforms: List[BaseTransform] = []
    all_transforms.extend(normalization_transforms)
    if pre_transforms:
        all_transforms.extend(list(pre_transforms.transforms))

    combined_transforms = Compose(all_transforms)
    logger.info(f"Prepared combined transforms for Pass 2")


    curated_data_list: List[Data] = []

    for chunk_file in tqdm(chunk_files, desc="Pass 2/2: Normalizing and Transforming"):
        try:
            chunk = torch.load(chunk_file)
            if not isinstance(chunk, list):
                logger.warning(f"Chunk file {chunk_file} did not load as a list of Data objects during transformation. Skipping.")
                raise InvalidDataInChunkError(
                    chunk_file=chunk_file,
                    item_index=-1,
                    message=f"Chunk file {chunk_file} loaded but is not a list during transformation."
                )
        except InvalidDataInChunkError:
            continue
        except Exception as e:
            logger.error(f"Error loading chunk file {chunk_file} during transformation: {e}. Skipping this chunk.")
            raise ChunkFileLoadError(
                file_path=chunk_file,
                message=f"Failed to load or parse chunk file {chunk_file} during transformation pass",
                original_exception=e
            )

        for i, data in enumerate(chunk):
            if not isinstance(data, Data):
                logger.warning(f"Item {i} in chunk {chunk_file} is not a PyG Data object. Skipping for transformation.")
                continue

            try:
                data = combined_transforms(data)
                curated_data_list.append(data)
            except (PreTransformApplicationError, FeatureNormalizationError) as e:
                logger.warning(f"Skipping graph from chunk {chunk_file}, index {i} due to transform error: {e}. Original Exception: {e.original_exception if hasattr(e, 'original_exception') else 'N/A'}")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred during transformation for graph in chunk {chunk_file}, index {i}: {e}. Skipping this graph.")
                raise GraphCurationError(
                    message=f"Unexpected error during graph transformation in chunk {chunk_file}, index {i}.",
                    original_exception=e
                ) from e


    if curated_data_list:
        try:
            torch.save(curated_data_list, output_path)
            logger.info(f"Curation complete. Curated dataset saved to: {output_path} with {len(curated_data_list)} graphs.")
        except Exception as e:
            logger.error(f"Error saving curated dataset to {output_path}: {e}")
            raise DatasetSaveError(
                output_path=output_path,
                message=f"Failed to save the final curated dataset to {output_path}.",
                original_exception=e
            )
    else:
        logger.warning("No graphs were successfully curated. Output file not created.")

    try:
        if chunk_dir.exists() and chunk_dir.is_dir():
            shutil.rmtree(chunk_dir)
            logger.info(f"Cleaned up intermediate chunk directory: {chunk_dir}")
        else:
            logger.debug(f"Chunk directory {chunk_dir} does not exist or is not a directory. No cleanup needed.")
    except OSError as e:
        logger.error(f"OS error cleaning up intermediate directory {chunk_dir}: {e}. Check permissions or if directory is in use.")
        raise ChunkDirectoryCleanupError(
            chunk_dir=chunk_dir,
            message=f"OS error during cleanup of {chunk_dir}.",
            original_exception=e
        ) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred cleaning up intermediate directory {chunk_dir}: {e}")
        raise ChunkDirectoryCleanupError(
            chunk_dir=chunk_dir,
            message=f"Unexpected error during cleanup of {chunk_dir}.",
            original_exception=e
        ) from e
