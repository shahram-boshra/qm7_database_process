# QM7 Dataset Processing and Curation to PyTorch Geometric Molecular Graphs

[![Python Version](https://img.shields.io/badge/python-%3E=3.8-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-brightgreen.svg)](https://peps.python.org/pep-0008/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%23167ac6)](https://pycqa.github.io/isort/)
[![Linter: pylint](https://img.shields.io/badge/linter-pylint-yellowgreen)](https://www.pylint.org/)
[![Formatter: black](https://img.shields.io/badge/formatter-black-000000?style=flat&logo=python&logoColor=yellow)](https://github.com/psf/black)



This repository provides a robust Python pipeline for processing the QM7 quantum chemistry dataset into a graph-based format, optimized for PyTorch Geometric (PyG). It handles the parsing of raw SDF, CSV, and MAT files, constructs detailed molecular graphs, and applies essential data curation steps including feature normalization and custom transformations.
Table of Contents

    Introduction
    Features
    Dataset
    Installation
    Usage
        Prepare Data Directories
        Example Workflow
    Project Structure
    Core Components
    Error Handling
    Contributing
    License
    Acknowledgments

Introduction

The QM7 dataset is a widely used benchmark in quantum chemistry, comprising 7165 molecules with up to 23 atoms (C, O, N, S, H) and their corresponding atomization energies. This project offers a comprehensive pipeline to transform this raw data (from SDF, MAT, and CSV files) into a structured graph format, where each molecule is represented as a torch_geometric.data.Data object. This structured format is ideal for applying Graph Neural Networks (GNNs) to tasks like molecular property prediction.

The processing pipeline covers:

    Loading molecular structures, atomic charges, Coulomb matrices, and target energies.
    Thorough data alignment and consistency checks across diverse data sources.
    Construction of PyTorch Geometric Data objects with rich node and edge features.
    Support for pre-filtering graphs based on various criteria (e.g., number of atoms, carbon count).
    Global feature normalization (mean-variance scaling) using sklearn.preprocessing.StandardScaler.
    Integration of custom PyG transforms.
    Efficiently saving the processed dataset in manageable chunks before consolidating it into a single .pt file.

Features

    Robust Data Loading: Seamlessly handles SDF, CSV, and .mat files with integrated error handling and logging.
    Data Consistency Checks: Aligns RDKit molecules with .mat data based on atomic composition, ensuring high data integrity.
    Rich Graph Representation: Each Data object includes:
        Node Features (x): One-hot encoded atom type, atomic number, aromaticity, hybridization (SP, SP2, SP3), total number of hydrogens, atomic charge (from QM7 .mat file), and diagonal element of the Coulomb matrix.
        Atomic Numbers (z): Pure atomic numbers for each node.
        3D Positions (pos): Atomic coordinates derived from the SDF conformers.
        Edge Features (edge_attr): One-hot encoded RDKit bond type (Single, Double, Triple, Aromatic) and the off-diagonal element of the Coulomb matrix.
        Target (y): Atomization energy.
        Original Index (idx): Retains the original index for full traceability.
    Modular Pre-filtering: Allows you to define and apply custom filters (e.g., by node count, specific atom counts) to efficiently exclude unwanted graphs early in the pipeline.
    Global Feature Normalization: Automatically calculates and applies StandardScaler transformations to specified node and edge features across the entire dataset.
    Customizable Transforms: Integrates effortlessly with torch_geometric.transforms.Compose for any additional user-defined graph transformations.
    Chunked Processing: Processes large datasets in a memory-efficient manner by saving intermediate results to disk in chunks.
    Comprehensive Error Handling: Custom exception classes provide detailed and actionable information for processing failures.
    Detailed Logging: Offers informative logs to monitor the entire processing pipeline.

Dataset

The QM7 dataset can typically be obtained from the following sources:

    GDB-9 related datasets: You'll need gdb7.sdf, atomization_energies.csv, and qm7.mat. You can often find these at https://figshare.com/articles/QM7_dataset/1930773.
    MoleculeNet's QM7 page: https://moleculenet.org/datasets/qm7

Required Files:

    gdb7.sdf: Contains molecular structures (RDKit Chem.Mol objects).
    atomization_energies.csv: Contains atomization energies.
    qm7.mat: Contains Coulomb matrices (X) and atomic charges (Z).

Placement: Please place these required files in a designated raw_data directory within your project, for example, data/qm7/raw_data.
Installation

    Clone the repository:
    Bash

git clone https://github.com/shahram-boshra/qm7_database_process.git
cd qm7_database_process # Make sure to change into the cloned directory

Create a virtual environment (recommended):
Bash

python -m venv venv
source venv/bin/activate # On Windows: `venv\Scripts\activate`

Install dependencies:
Bash

    pip install -r requirements.txt

    The requirements.txt file should contain the following:

    torch
    torch_geometric
    rdkit-pypi
    numpy
    scipy
    pandas
    tqdm
    scikit-learn
    PyYAML

Usage
Prepare Data Directories

First, ensure your raw QM7 dataset files are organized as follows:

.
├── data/
│   └── qm7/
│       └── raw_data/                 # Directory for your original QM7 dataset files
│           ├── gdb7.sdf
│           ├── atomization_energies.csv
│           └── qm7.mat
│       └── processed/                # Output directory for the final curated data
└── scripts/
    ├── pyg_qm7_processing.py
    ├── qm7_curation.py
    ├── exceptions.py
    └── main_process.py               # Example script to run the pipeline

Example Workflow

Below is an example of how you can integrate and use the processing pipeline. You would typically create a main_process.py (or similar) script to orchestrate these steps.
Python

# main_process.py
import logging
from pathlib import Path
import torch
from functools import partial
from torch_geometric.transforms import Compose

# Assuming these are in your project's scripts directory or properly installed
from pyg_qm7_processing import process_qm7_data, filter_by_max_nodes, filter_by_min_nodes, filter_by_num_carbons
from qm7_curation import curate_qm7_data, CustomEdgeFeatureCombiner
from exceptions import PipelineError, GraphCurationError, QM7SpecificProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_qm7_pipeline():
    # Define paths
    base_data_dir = Path("data/qm7")
    raw_data_dir = base_data_dir / "raw_data"
    processed_data_dir = base_data_dir / "processed"
    intermediate_chunk_dir = base_data_dir / "intermediate_chunks"

    sdf_file = raw_data_dir / "gdb7.sdf"
    energies_file = raw_data_dir / "atomization_energies.csv"
    mat_file = raw_data_dir / "qm7.mat"
    final_output_file = processed_data_dir / "qm7_processed_normalized.pt"

    # Create directories if they don't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    intermediate_chunk_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Initial Processing into Chunks ---
    chunk_size = 1000 # Number of graphs per chunk file

    # Define pre-filters (optional)
    # This example filters for graphs with 5 to 20 nodes, and at least 1 carbon
    combined_pre_filter = lambda data: (
        filter_by_min_nodes(data, min_nodes=5) and
        filter_by_max_nodes(data, max_nodes=20) and
        filter_by_num_carbons(data, min_carbons=1)
    )

    try:
        logger.info("--- Starting initial QM7 data processing into chunks ---")
        temp_chunk_dir_path = process_qm7_data(
            sdf_file=sdf_file,
            energies_file=energies_file,
            mat_file=mat_file,
            intermediate_chunk_output_dir=intermediate_chunk_dir,
            chunk_size=chunk_size,
            pre_filter=combined_pre_filter
        )
        logger.info(f"Initial processing complete. Chunks saved to: {temp_chunk_dir_path}")
    except QM7SpecificProcessingError as e:
        logger.exception(f"Initial QM7 processing failed due to: {e}")
        return
    except Exception as e:
        logger.exception(f"An unexpected error occurred during initial QM7 processing: {e}")
        return

    # --- Step 2: Curation (Normalization and Final Transforms) ---
    feature_keys_to_normalize = ['x', 'edge_attr'] # Features to apply StandardScaler to

    # Define additional pre-transforms to apply after normalization (optional)
    # Note: `CustomEdgeFeatureCombiner` is defined in qm7_curation.py.
    # If you have other custom transforms, ensure they are imported.
    additional_transforms = Compose([
        CustomEdgeFeatureCombiner(param1='value1', param2='value2'),
        # Add more transforms here if needed, e.g.,
        # T.ToSparseTensor(),
    ])

    try:
        logger.info("--- Starting QM7 data curation (normalization and final transforms) ---")
        curate_qm7_data(
            chunk_dir=temp_chunk_dir_path,
            output_path=final_output_file,
            feature_keys_for_norm=feature_keys_to_normalize,
            pre_transforms=additional_transforms # These are applied *after* normalization
        )
        logger.info(f"QM7 data curation complete. Final dataset saved to: {final_output_file}")
    except GraphCurationError as e:
        logger.exception(f"QM7 data curation failed due to: {e}")
    except PipelineError as e: # Catching the base exception for broader errors
        logger.exception(f"QM7 data curation failed due to a pipeline error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during QM7 data curation: {e}")

if __name__ == "__main__":
    run_qm7_pipeline()

To run this example:

    Save the code above as main_process.py in your project's scripts directory.
    Ensure pyg_qm7_processing.py, qm7_curation.py, and exceptions.py are in the same directory or accessible in your Python path.
    Place the raw QM7 dataset files (gdb7.sdf, atomization_energies.csv, qm7.mat) in data/qm7/raw_data/.
    Execute the script:
    Bash

    python scripts/main_process.py

    This will process the data, save intermediate chunks, curate them (including normalization), save the final dataset, and clean up the intermediate chunks.

##Project Structure

.
├── data/
│   └── qm7/
│       ├── raw_data/               # Original QM7 dataset files (gdb7.sdf, atomization_energies.csv, qm7.mat)
│       ├── intermediate_chunks/    # Temporary directory for chunked PyG Data objects
│       └── processed/              # Output directory for the final curated dataset
├── scripts/
│   ├── pyg_qm7_processing.py       # Core script for initial data loading, graph construction, and chunking
│   ├── qm7_curation.py             # Script for global statistics calculation, normalization, and final transforms
│   ├── exceptions.py               # Custom exception definitions for robust error handling
│   └── main_process.py             # Example script to run the end-to-end processing pipeline
├── .gitignore
├── requirements.txt
└── README.md

Core Components
pyg_qm7_processing.py

This module manages the initial phase of data processing, including:

    load_sdf_molecules: Loads RDKit molecule objects from an SDF file.
    load_atomization_energies_pandas: Loads target energies from a CSV file.
    load_qm7_mat_data: Loads Coulomb matrices and atomic charges from a .mat file.
    identify_sdf_indices_to_exclude: Performs crucial consistency checks between SDF and MAT data (based on atomic composition) to identify and exclude inconsistent entries.
    create_pyg_graph: Constructs a torch_geometric.data.Data object for a single molecule, integrating all features.
    filter_by_max_nodes, filter_by_min_nodes, filter_by_num_carbons: Example pre-filter functions to exclude graphs based on criteria like number of nodes or specific atom counts.
    process_qm7_data: The main orchestration function for this module, responsible for loading all raw data, applying consistency checks and pre-filters, and saving the processed graphs into memory-efficient chunks.

qm7_curation.py

This module handles the post-processing and curation of the chunked data:

    calculate_global_stats_with_scaler: Conducts the first pass over the chunked data to compute global mean and standard deviation for specified features using sklearn.preprocessing.StandardScaler.partial_fit.
    StandardScalerTransform: A custom torch_geometric.transforms.BaseTransform that applies a pre-fitted StandardScaler to a specified feature of a PyG Data object.
    CustomEdgeFeatureCombiner: A placeholder custom PyG transform demonstrating how to incorporate user-defined transformations into the pipeline. You can modify this to implement specific feature engineering for edge attributes.
    curate_qm7_data: The main orchestration function for this module, which performs a second pass over the data. It applies the fitted StandardScalerTransform for normalization, applies any additional pre_transforms (e.g., CustomEdgeFeatureCombiner), concatenates all processed graphs, and saves the final dataset to a single .pt file. It also cleans up the intermediate chunk directory.

exceptions.py

This module defines custom exception classes, providing more specific and informative error messages throughout the data processing pipeline. This significantly aids in debugging and quickly identifying and resolving issues.
Error Handling

The codebase is designed with robust error handling using custom exception classes defined in exceptions.py. Each critical step (file loading, data alignment, graph construction, feature extraction, normalization, saving) is wrapped in try-except blocks. This ensures that specific issues are caught, and clear, actionable error messages are provided, which greatly assists in rapidly identifying and resolving problems during data processing.
##Contributing

Contributions are welcome! If you encounter a bug or have suggestions for improvements, please feel free to open an issue or submit a pull request.
##License

This project is open-source and available under the MIT License.
##Acknowledgments

We gratefully acknowledge the contributions of the following:

    The QM7 dataset providers for making this valuable quantum chemistry benchmark dataset publicly available.
    The developers of RDKit, an open-source cheminformatics software, for providing robust tools essential for molecular manipulation and feature extraction in this project.
    The PyTorch Geometric team for their flexible and powerful library, which significantly simplified the implementation of graph neural network functionalities.
    The PyTorch development team for their comprehensive deep learning framework, foundational to our tensor operations and overall computational graph.
    The creators of NumPy and Pandas for their indispensable libraries, crucial for efficient numerical computation and data handling.
    The scikit-learn community for their high-quality machine learning tools, particularly the StandardScaler used for feature normalization.
