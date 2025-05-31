# db_dl.py

import os
import requests
from pathlib import Path
import tarfile
import logging
import sys
from typing import Union

from exceptions import (
    DownloadError,
    ExtractionError,
    DirectoryCreationError,
    MissingDownloadURLError,
    DownloadSetupError
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path) -> None:
    """
    Downloads a file from a given URL to a specified output path.

    Args:
        url (str): The URL of the file to download.
        output_path (Path): The local path where the file should be saved.

    Raises:
        DownloadError: If the download fails due to network issues, HTTP errors,
                        or unexpected errors during file writing.
    """
    if output_path.exists():
        logger.info(f"File already exists: {output_path}. Skipping download.")
        return

    logger.info(f"Downloading {url} to {output_path}...")
    try:
        response: requests.Response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as file:
            for data in response.iter_content(chunk_size=8192):
                file.write(data)

        logger.info(f"Successfully downloaded {output_path}")

    except requests.exceptions.RequestException as e:
        raise DownloadError(
            url=url,
            path=output_path,
            message=f"Download failed for {url}: {e}",
            original_exception=e
        ) from e
    except Exception as e:
        raise DownloadError(
            url=url,
            path=output_path,
            message=f"An unexpected error occurred during download of {url}: {e}",
            original_exception=e
        ) from e


def extract_tar_gz(file_path: Path, extract_dir: Path) -> None:
    """
    Extracts a .tar.gz archive to a specified directory.

    Args:
        file_path (Path): The path to the .tar.gz archive file.
        extract_dir (Path): The directory where the contents should be extracted.

    Raises:
        ExtractionError: If the archive file is not found, is corrupted,
                          or if an extraction error occurs (including path traversal attempts).
    """
    if not file_path.exists():
        raise ExtractionError(
            file_path=file_path,
            message=f"Archive file not found: {file_path}. Cannot extract."
        )

    expected_sdf: Path = extract_dir / "gdb7.sdf"
    expected_csv: Path = extract_dir / "gdb7.sdf.csv"
    if expected_sdf.exists() and expected_csv.exists():
        logger.info(f"Expected extracted files ({expected_sdf.name}, {expected_csv.name}) already exist in {extract_dir}. Skipping extraction of {file_path.name}.")
        return

    logger.info(f"Extracting {file_path} to {extract_dir}...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            def is_within_directory(directory: Union[str, Path], target: Union[str, Path]) -> bool:
                """
                Checks if a target path is within the specified directory to prevent path traversal.

                Args:
                    directory (Union[str, Path]): The base directory.
                    target (Union[str, Path]): The target path to check.

                Returns:
                    bool: True if the target path is within the directory, False otherwise.
                """
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            members: list[tarfile.TarInfo] = tar.getmembers()
            for member in members:
                member_path: Path = (extract_dir / member.name).resolve()
                if not is_within_directory(extract_dir, member_path):
                    raise ExtractionError(
                        file_path=file_path,
                        extract_dir=extract_dir,
                        message=f"Attempted Path Traversal in Tar File for member: {member.name}. Extraction aborted.",
                        original_exception=ValueError("Path traversal attempt detected")
                    )

            tar.extractall(path=extract_dir)
        logger.info(f"Successfully extracted {file_path.name}")
    except tarfile.ReadError as e:
        raise ExtractionError(
            file_path=file_path,
            message=f"Error reading tar.gz file {file_path}: {e}. It might be corrupted or not a gzipped tar file.",
            original_exception=e
        ) from e
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(
            file_path=file_path,
            extract_dir=extract_dir,
            message=f"An unexpected error occurred during extraction of {file_path}: {e}",
            original_exception=e
        ) from e


def main(base_data_dir: Path, qm7_sdf_url: str, qm7_mat_url: str) -> None:
    """
    Main function to orchestrate the download and initial organization of QM7 data.
    All parameters are expected to be provided by the calling script (e.g., main.py).

    Args:
        base_data_dir (Path): The base directory for data storage.
        qm7_sdf_url (str): The URL for the QM7 SDF tar.gz file.
        qm7_mat_url (str): The URL for the QM7 MAT file.

    Raises:
        DownloadSetupError: If any critical error occurs during directory creation,
                            downloading, or extraction.
        MissingDownloadURLError: If any of the required URLs are empty strings or None.
    """
    logger.info("--- Starting QM7 Data Download and Setup ---")

    if not base_data_dir:
        raise DownloadSetupError(message="base_data_dir cannot be empty or None.")
    if not qm7_sdf_url:
        raise MissingDownloadURLError(url_key="qm7_sdf_url", message="The URL for QM7 SDF is missing.")
    if not qm7_mat_url:
        raise MissingDownloadURLError(url_key="qm7_mat_url", message="The URL for QM7 MAT is missing.")

    qm7_data_dir: Path = base_data_dir / "qm7"
    raw_dir: Path = qm7_data_dir / "raw"
    processed_dir: Path = qm7_data_dir / "processed"

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured raw data directory exists: {raw_dir}")
        logger.info(f"Ensured processed data directory exists: {processed_dir}")
    except OSError as e:
        raise DirectoryCreationError(
            path=f"{raw_dir} or {processed_dir}",
            message=f"Failed to create data directories: {e}",
            original_exception=e
        ) from e
    except Exception as e:
        raise DownloadSetupError(
            message=f"An unexpected error occurred during directory creation: {e}",
            original_exception=e
        ) from e

    gdb7_tar_gz_path: Path = qm7_data_dir / "gdb7.sdf.tar.gz"
    qm7_mat_path: Path = raw_dir / "qm7.mat"

    try:
        download_file(qm7_sdf_url, gdb7_tar_gz_path)
        download_file(qm7_mat_url, qm7_mat_path)
    except DownloadError as e:
        raise DownloadSetupError(
            message=f"Data download failed: {e}",
            original_exception=e
        ) from e
    except Exception as e:
        raise DownloadSetupError(
            message=f"An unexpected error occurred during data download: {e}",
            original_exception=e
        ) from e

    try:
        extract_tar_gz(gdb7_tar_gz_path, raw_dir)
    except ExtractionError as e:
        raise DownloadSetupError(
            message=f"Data extraction failed: {e}",
            original_exception=e
        ) from e
    except Exception as e:
        raise DownloadSetupError(
            message=f"An unexpected error occurred during data extraction: {e}",
            original_exception=e
        ) from e

    logger.info("--- QM7 Data Download and Setup Complete ---")
