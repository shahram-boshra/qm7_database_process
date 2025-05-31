# pyg_qm7_processing.py

import logging
from pathlib import Path
import torch
from torch_geometric.data import Data
import torch_geometric.data.data
import torch_geometric.data.storage
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
import numpy as np
import scipy.io
import pandas as pd
from tqdm import tqdm
from collections import Counter
import shutil
from functools import partial
from typing import List, Dict, Tuple, Set, Union, Optional, Callable, Any

from exceptions import (
    QM7SpecificProcessingError, QM7DataLoadError, QM7DataAlignmentError,
    QM7GraphConstructionError, QM7InvalidMoleculeError, QM7FeatureExtractionError,
    QM7EdgeFeatureMismatchError, QM7ChunkSavingError, QM7PreFilterError,
    FileSystemError, DataLoadingError
)


logger = logging.getLogger(__name__)

QM7_ATOM_TYPES: Dict[str, int] = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4}
NUM_ATOM_TYPES: int = len(QM7_ATOM_TYPES)

BOND_TYPES: Dict[BondType, int] = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
NUM_BOND_TYPES: int = len(BOND_TYPES)

def load_sdf_molecules(sdf_path: Path) -> List[Chem.Mol]:
    """
    Loads molecules from an SDF file using RDKit's `SDMolSupplier`.

    This function provides robust loading of SDF files, including checks for
    file existence and emptiness. It iterates through the supplier to identify
    and log any individual molecules that RDKit fails to parse, skipping them
    and continuing to load valid molecules.

    Args:
        sdf_path (Path): The file path to the SDF file.

    Returns:
        List[Chem.Mol]: A list of RDKit molecule objects successfully loaded
                        from the SDF file. Returns an empty list if the file
                        is empty or no molecules could be loaded.

    Raises:
        QM7DataLoadError: If the SDF file is not found, is empty, or if an
                          unexpected error occurs during the loading process.
    """
    try:
        if not sdf_path.exists():
            logger.error(f"SDF file not found at {sdf_path}. Please check the path.")
            raise QM7DataLoadError(sdf_path, "SDF", message=f"SDF file not found: {sdf_path}")

        if sdf_path.stat().st_size == 0:
            logger.warning(f"SDF file {sdf_path} is empty. Returning an empty list of molecules.")
            return []

        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        molecules: List[Chem.Mol] = []
        for i, mol in enumerate(suppl):
            if mol is not None:
                molecules.append(mol)
            else:
                logger.warning(f"RDKit failed to load molecule at index {i} from {sdf_path}. Skipping this molecule.")

        logger.info(f"Loaded {len(molecules)} molecules from {sdf_path}")
        return molecules
    except FileNotFoundError as e:
        logger.error(f"Error: SDF file not found at {sdf_path}: {e}")
        raise QM7DataLoadError(sdf_path, "SDF", message=f"SDF file not found: {sdf_path}", original_exception=e) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading SDF file {sdf_path}: {e}")
        raise QM7DataLoadError(sdf_path, "SDF", message=f"Unexpected error loading SDF file {sdf_path}", original_exception=e) from e


def load_atomization_energies_pandas(csv_path: Path) -> np.ndarray:
    """
    Loads atomization energies from a QM7 CSV file using pandas.

    This function specifically targets the QM7 CSV format, expecting energies
    in the first column and skipping the header row 'u0_atom'. It performs
    checks for file existence, emptiness, and potential parsing errors to
    ensure robust data loading.

    Args:
        csv_path (Path): The file path to the CSV file containing atomization energies.

    Returns:
        np.ndarray: A NumPy array of float32 representing the atomization energies.
                    Returns an empty NumPy array if the file is empty or contains
                    no data after skipping the header.

    Raises:
        QM7DataLoadError: If the CSV file is not found, is empty or malformed,
                          or if the expected column is missing.
    """
    try:
        if not csv_path.exists():
            logger.error(f"CSV file not found at {csv_path}. Please check the path.")
            raise QM7DataLoadError(csv_path, "CSV", message=f"CSV file not found: {csv_path}")

        if csv_path.stat().st_size == 0:
            logger.warning(f"CSV file {csv_path} is empty. Returning an empty NumPy array as per original behavior.")
            return np.array([], dtype=np.float32)

        df = pd.read_csv(csv_path, header=None, names=['atomization_energy'], skiprows=1)

        if df.empty:
            logger.info(f"CSV file {csv_path} loaded successfully but contains no data after skipping header. Returning an empty NumPy array.")
            return np.array([], dtype=np.float32)

        energies: np.ndarray = df['atomization_energy'].to_numpy(dtype=np.float32)

        if energies.size == 0:
            logger.info(f"No atomization energies found in {csv_path} after processing. Returning an empty NumPy array.")
            return np.array([], dtype=np.float32)

        logger.info(f"Loaded {len(energies)} atomization energies from {csv_path} using pandas.")
        return energies
    except FileNotFoundError as e:
        logger.error(f"Error: CSV file not found at {csv_path}: {e}")
        raise QM7DataLoadError(csv_path, "CSV", message=f"CSV file not found: {csv_path}", original_exception=e) from e
    except pd.errors.EmptyDataError as e:
        logger.error(f"Error: CSV file {csv_path} is empty or only contains a header and no data after skipping: {e}. Raising QM7DataLoadError.")
        raise QM7DataLoadError(csv_path, "CSV", message=f"CSV file empty or malformed header: {csv_path}", original_exception=e) from e
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {csv_path}. It might be malformed: {e}. Raising QM7DataLoadError.")
        raise QM7DataLoadError(csv_path, "CSV", message=f"Error parsing CSV file {csv_path}. Malformed.", original_exception=e) from e
    except KeyError as e:
        logger.error(f"Error: 'atomization_energy' column not found in {csv_path}. Check CSV format and skiprows: {e}. Raising QM7DataLoadError.")
        raise QM7DataLoadError(csv_path, "CSV", message=f"Missing 'atomization_energy' column in {csv_path}", original_exception=e) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading atomization energies from {csv_path}: {e}. Raising QM7DataLoadError.")
        raise QM7DataLoadError(csv_path, "CSV", message=f"Unexpected error loading CSV file {csv_path}", original_exception=e) from e


def load_qm7_mat_data(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads Coulomb matrices and atomic charges from the QM7 `.mat` file.

    This function expects the `.mat` file to contain two specific keys:
    'X' for Coulomb matrices and 'Z' for atomic charges. It validates file
    existence, checks for emptiness, and ensures the presence of these keys.

    Args:
        mat_path (Path): The file path to the QM7 `.mat` file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                        - The Coulomb matrices (shape `N x M x M`, float32).
                                        - The atomic charges (shape `N x M`, float32).
                                        where N is the number of molecules and M is
                                        the maximum number of atoms in a molecule.

    Raises:
        QM7DataLoadError: If the .mat file is not found, is empty, or does not
                          contain the expected 'X' or 'Z' keys, or if arrays are empty.
    """
    try:
        if not mat_path.exists():
            logger.error(f".mat file not found at {mat_path}. Please check the path.")
            raise QM7DataLoadError(mat_path, "MAT", message=f"MAT file not found: {mat_path}")

        if mat_path.stat().st_size == 0:
            logger.warning(f".mat file {mat_path} is empty. Raising QM7DataLoadError.")
            raise QM7DataLoadError(mat_path, "MAT", message=f"MAT file is empty: {mat_path}")

        mat_data: Dict[str, Any] = scipy.io.loadmat(str(mat_path))

        if 'X' not in mat_data:
            logger.error(f"Key 'X' not found in .mat file {mat_path}. It might be corrupted or in an unexpected format.")
            raise QM7DataLoadError(mat_path, "MAT", message=f"Missing 'X' key in MAT file {mat_path}")
        if 'Z' not in mat_data:
            logger.error(f"Key 'Z' not found in .mat file {mat_path}. It might be corrupted or in an unexpected format.")
            raise QM7DataLoadError(mat_path, "MAT", message=f"Missing 'Z' key in MAT file {mat_path}")

        x: np.ndarray = mat_data['X'].astype(np.float32)
        z: np.ndarray = mat_data['Z'].astype(np.float32)

        if x.size == 0 or z.size == 0:
            logger.warning(f"Loaded .mat file {mat_path} but 'X' or 'Z' arrays are empty. X shape: {x.shape}, Z shape: {z.shape}. Raising QM7DataLoadError.")
            raise QM7DataLoadError(mat_path, "MAT", message=f"Empty 'X' or 'Z' arrays in MAT file {mat_path}")

        logger.info(f"Loaded X (shape: {x.shape}) and Z (shape: {z.shape}) from {mat_path}")
        return x, z
    except FileNotFoundError as e:
        logger.error(f"Error: .mat file not found at {mat_path}: {e}")
        raise QM7DataLoadError(mat_path, "MAT", message=f"MAT file not found: {mat_path}", original_exception=e) from e
    except KeyError as e:
        logger.error(f"Error: Expected key {e} not found in .mat file {mat_path}. File might be corrupted or malformed. Raising QM7DataLoadError.")
        raise QM7DataLoadError(mat_path, "MAT", message=f"Missing expected key in MAT file {mat_path}", original_exception=e) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading .mat file {mat_path}: {e}")
        raise QM7DataLoadError(mat_path, "MAT", message=f"Unexpected error loading MAT file {mat_path}", original_exception=e) from e


def _get_atom_composition_from_mol(mol: Optional[Chem.Mol]) -> Counter:
    """
    Helper function to extract the atom composition (symbol and count)
    from a single RDKit molecule object.

    Args:
        mol (Optional[Chem.Mol]): An RDKit molecule object. Can be `None`.

    Returns:
        Counter: A `collections.Counter` object where keys are atom symbols
                 (e.g., 'C', 'H') and values are their respective counts in the molecule.
                 Returns an empty Counter if `mol` is `None` or has no atoms.
    """
    if mol is None:
        return Counter()
    atomic_symbols: List[str] = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Counter(atomic_symbols)

def _get_atom_composition_from_mat_z(mat_z_entry: np.ndarray) -> Counter:
    """
    Helper function to extract the atom composition (symbol and count)
    from a single entry of the atomic charges array (`Z`) loaded from the `.mat` file.

    It converts atomic numbers (integers) to atomic symbols using RDKit's
    periodic table and ignores zero values, which typically represent padding.

    Args:
        mat_z_entry (np.ndarray): A 1D NumPy array representing the atomic
                                  numbers for a single molecule, typically
                                  an entry from the 'Z' array of the QM7 .mat file.

    Returns:
        Counter: A `collections.Counter` object where keys are atom symbols
                 (e.g., 'C', 'H') and values are their respective counts.
                 Returns an empty Counter if `mat_z_entry` contains only zeros
                 or is empty.
    """
    atomic_symbols: List[str] = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in mat_z_entry if z != 0]
    return Counter(atomic_symbols)

def identify_sdf_indices_to_exclude(sdf_molecules: List[Chem.Mol], mat_atomic_charges_data: np.ndarray) -> Set[int]:
    """
    Identifies indices of RDKit molecules from the SDF file that should be
    excluded from further processing due to inconsistencies with the atomic
    charges data (`Z` array) loaded from the `.mat` file.

    This function performs a two-criterion consistency check:
    1.  **Composition Presence**: Checks if the exact atom composition (types and counts)
        of an SDF molecule is present anywhere within the set of compositions
        derived from the MAT `Z` data.
    2.  **Composition Count**: Compares the frequency of each unique atom composition
        between the SDF molecules and the MAT `Z` data. Molecules are excluded
        if their composition cannot be matched one-to-one with available
        compositions from the MAT data, accounting for duplicates.

    This process helps align the two datasets and filter out potentially malformed
    or misaligned entries.

    Args:
        sdf_molecules (List[Chem.Mol]): A list of RDKit molecule objects loaded from the SDF file.
        mat_atomic_charges_data (np.ndarray): The NumPy array of atomic charges
                                            ('Z' array) loaded from the QM7 `.mat` file.
                                            Shape is typically `(N, M)`, where N is the
                                            number of molecules and M is max atoms.

    Returns:
        Set[int]: A set of integer indices corresponding to the SDF molecules
                  that were identified as inconsistent and should be excluded.
    """
    logger.info("\n--- Starting In-Process Consistency Check ---")

    sdf_atom_data_full: List[Dict[str, Union[int, Counter]]] = [{'index': i, 'atoms': _get_atom_composition_from_mol(mol)}
                                  for i, mol in enumerate(sdf_molecules)]
    mat_atom_data_full: List[Dict[str, Union[int, Counter]]] = [{'index': i, 'atoms': _get_atom_composition_from_mat_z(mat_z_entry)}
                                  for i, mat_z_entry in enumerate(mat_atomic_charges_data)]

    sdf_compositions_set: Set[Tuple[Tuple[str, int], ...]] = {tuple(sorted(entry['atoms'].items())) for entry in sdf_atom_data_full}
    mat_compositions_set: Set[Tuple[Tuple[str, int], ...]] = {tuple(sorted(entry['atoms'].items())) for entry in mat_atom_data_full}

    sdf_indices_no_presence_match: Set[int] = set()
    for entry in sdf_atom_data_full:
        composition_tuple: Tuple[Tuple[str, int], ...] = tuple(sorted(entry['atoms'].items()))
        if composition_tuple not in mat_compositions_set:
            sdf_indices_no_presence_match.add(entry['index'])

    if sdf_indices_no_presence_match:
        logger.warning(f"SDF indices whose atom composition is NOT present in MAT file (Criterion 1): {sorted(list(sdf_indices_no_presence_match))}")

    sdf_composition_counts: Counter = Counter(tuple(sorted(entry['atoms'].items())) for entry in sdf_atom_data_full)
    mat_composition_counts: Counter = Counter(tuple(sorted(entry['atoms'].items())) for entry in mat_atom_data_full)

    sdf_indices_by_count_mismatch: Set[int] = set()
    temp_mat_composition_counts: Counter = mat_composition_counts.copy()

    for entry in sdf_atom_data_full:
        sdf_idx: int = entry['index']
        composition_tuple: Tuple[Tuple[str, int], ...] = tuple(sorted(entry['atoms'].items()))

        if temp_mat_composition_counts.get(composition_tuple, 0) > 0:
            temp_mat_composition_counts[composition_tuple] -= 1
        else:
            sdf_indices_by_count_mismatch.add(sdf_idx)

    if sdf_indices_by_count_mismatch:
        logger.warning(f"SDF indices that cannot be matched by composition count (Criterion 2): {sorted(list(sdf_indices_by_count_mismatch))}")

    final_indices_to_exclude: Set[int] = sdf_indices_no_presence_match.union(sdf_indices_by_count_mismatch)
    logger.info(f"Total SDF indices to exclude after comprehensive consistency check: {sorted(list(final_indices_to_exclude))}")
    logger.info("--- Consistency Check Complete ---\n")

    return final_indices_to_exclude

def create_pyg_graph(mol: Chem.Mol, atom_charges: np.ndarray, coulomb_matrix: np.ndarray, target_energy: float, original_idx: int) -> Data:
    """
    Creates a PyTorch Geometric (PyG) `Data` object from an RDKit molecule
    and its associated QM7 dataset features (atomic charges, Coulomb matrix,
    and target energy).

    This function extracts various molecular properties and converts them
    into the tensor formats required by PyG, forming the nodes, edges,
    and their respective features, along with 3D positions and the target value.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        atom_charges (np.ndarray): A 1D NumPy array of atomic charges for the
                                   atoms in `mol`, derived from the QM7 `.mat` file.
        coulomb_matrix (np.ndarray): The Coulomb matrix for the molecule, derived
                                       from the QM7 `.mat` file.
        target_energy (float): The atomization energy of the molecule, serving as
                               the target property `y` for the graph.
        original_idx (int): The original index of the molecule in the full dataset,
                            stored as a `data.idx` attribute for traceability.

    Returns:
        Data: A PyTorch Geometric `Data` object representing the molecule,
              with `x` (node features), `z` (atomic numbers), `pos` (3D positions),
              `edge_index` (connectivity), `edge_attr` (edge features), and
              `y` (target energy).

    Raises:
        QM7InvalidMoleculeError: If `mol` is None or has no atoms or conformers.
        QM7GraphConstructionError: If there's a general error during graph
                                   construction (e.g., issues with conformer
                                   positions, target tensor conversion, or
                                   PyG Data object instantiation).
        QM7FeatureExtractionError: If there's an error while extracting specific
                                   features (e.g., atomic numbers, RDKit bond
                                   features, node features, or Coulomb matrix
                                   edge features).
        QM7EdgeFeatureMismatchError: If the number of RDKit-derived edges does
                                     not match the number of Coulomb matrix-derived
                                     edges.
    """
    if mol is None:
        logger.error(f"Attempted to create graph for original_idx {original_idx} from a None RDKit molecule.")
        raise QM7InvalidMoleculeError(original_idx, reason="RDKit molecule is None.")

    if mol.GetNumAtoms() == 0:
        logger.error(f"Molecule for original_idx {original_idx} has no atoms. Skipping graph creation.")
        raise QM7InvalidMoleculeError(original_idx, reason="Molecule has no atoms.")

    if not mol.GetConformers():
        logger.error(f"Molecule for original_idx {original_idx} has no conformer. Skipping graph creation.")
        raise QM7InvalidMoleculeError(original_idx, reason="Molecule has no conformer.")

    try:
        pos: torch.Tensor = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    except Exception as e:
        logger.error(f"Error getting conformer positions for molecule {original_idx}: {e}")
        raise QM7GraphConstructionError(original_idx, message=f"Error with conformer positions.", original_exception=e) from e

    try:
        atomic_numbers: List[int] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        z: torch.Tensor = torch.tensor(atomic_numbers, dtype=torch.long)
    except Exception as e:
        logger.error(f"Error extracting atomic numbers for molecule {original_idx}: {e}")
        raise QM7FeatureExtractionError(original_idx, "atomic_numbers", message=f"Error extracting atomic numbers.", original_exception=e) from e

    row: List[int] = []
    col: List[int] = []
    edge_attr_list: List[int] = []
    try:
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_type: BondType = bond.GetBondType()
            edge_attr_list.append(BOND_TYPES.get(bond_type, NUM_BOND_TYPES))
            edge_attr_list.append(BOND_TYPES.get(bond_type, NUM_BOND_TYPES))

        if not row:
            edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_rdkit: torch.Tensor = torch.empty((0, NUM_BOND_TYPES + 1), dtype=torch.float)
        else:
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr_rdkit = torch.nn.functional.one_hot(torch.tensor(edge_attr_list), num_classes=NUM_BOND_TYPES + 1).float()
    except Exception as e:
        logger.error(f"Error processing bonds/edge features for molecule {original_idx}: {e}")
        raise QM7FeatureExtractionError(original_idx, "rdkit_bond_features", message=f"Error processing RDKit bond features.", original_exception=e) from e

    atom_features_list: List[torch.Tensor] = []
    try:
        for i_atom, atom in enumerate(mol.GetAtoms()):
            atom_symbol: str = atom.GetSymbol()
            type_idx: int = QM7_ATOM_TYPES.get(atom_symbol, NUM_ATOM_TYPES)
            hybridization: HybridizationType = atom.GetHybridization()
            is_aromatic: int = 1 if atom.GetIsAromatic() else 0
            num_h: int = atom.GetTotalNumHs()

            if i_atom >= len(atom_charges) or i_atom >= coulomb_matrix.shape[0]:
                logger.error(f"Index out of bounds for atom_charges or coulomb_matrix at atom {i_atom} for molecule {original_idx}.")
                raise QM7FeatureExtractionError(original_idx, "node_features", message=f"Atom index {i_atom} out of bounds for charges/coulomb matrix.", original_exception=None)

            atom_feature: List[torch.Tensor] = [
                torch.nn.functional.one_hot(torch.tensor([type_idx]), num_classes=NUM_ATOM_TYPES + 1).squeeze(0).float(),
                torch.tensor([atom.GetAtomicNum()], dtype=torch.float),
                torch.tensor([is_aromatic], dtype=torch.float),
                torch.tensor([1 if hybridization == HybridizationType.SP else 0], dtype=torch.float),
                torch.tensor([1 if hybridization == HybridizationType.SP2 else 0], dtype=torch.float),
                torch.tensor([1 if hybridization == HybridizationType.SP3 else 0], dtype=torch.float),
                torch.tensor([num_h], dtype=torch.float),
                torch.tensor([atom_charges[i_atom]], dtype=torch.float),
                torch.tensor([coulomb_matrix[i_atom, i_atom]], dtype=torch.float)
            ]
            atom_features_list.append(torch.cat(atom_feature))

        if not atom_features_list:
            logger.warning(f"No atom features generated for molecule {original_idx}. This indicates an issue.")
            raise QM7FeatureExtractionError(original_idx, "node_features", message=f"No atom features generated.")

        x: torch.Tensor = torch.stack(atom_features_list)
    except IndexError as ie:
        logger.error(f"Indexing error during node feature creation for molecule {original_idx}: {ie}")
        raise QM7FeatureExtractionError(original_idx, "node_features", message=f"Indexing error during node feature creation.", original_exception=ie) from ie
    except Exception as e:
        logger.error(f"Error creating node features for molecule {original_idx}: {e}")
        raise QM7FeatureExtractionError(original_idx, "node_features", message=f"Error creating node features.", original_exception=e) from e

    bond_features_cm_list: List[torch.Tensor] = []
    try:
        if edge_index.shape[1] > 0:
            for i_edge in range(edge_index.shape[1]):
                u, v = edge_index[0, i_edge].item(), edge_index[1, i_edge].item()
                if u >= coulomb_matrix.shape[0] or v >= coulomb_matrix.shape[1]:
                    logger.error(f"Edge index ({u}, {v}) out of bounds for coulomb_matrix (shape {coulomb_matrix.shape}) for molecule {original_idx}.")
                    raise QM7FeatureExtractionError(original_idx, "cm_edge_features", message=f"Edge index ({u}, {v}) out of bounds for Coulomb matrix.", original_exception=None)
                bond_features_cm_list.append(torch.tensor([coulomb_matrix[u, v]], dtype=torch.float).unsqueeze(0))

            if not bond_features_cm_list:
                edge_attr_cm: torch.Tensor = torch.empty((0, 1), dtype=torch.float)
            else:
                edge_attr_cm = torch.cat(bond_features_cm_list, dim=0)
        else:
            edge_attr_cm = torch.empty((0, 1), dtype=torch.float)
    except IndexError as ie:
        logger.error(f"Indexing error during Coulomb matrix edge feature creation for molecule {original_idx}: {ie}")
        raise QM7FeatureExtractionError(original_idx, "cm_edge_features", message=f"Indexing error during CM edge feature creation.", original_exception=ie) from ie
    except Exception as e:
        logger.error(f"Error creating Coulomb matrix edge features for molecule {original_idx}: {e}")
        raise QM7FeatureExtractionError(original_idx, "cm_edge_features", message=f"Error creating CM edge features.", original_exception=e) from e

    try:
        if edge_attr_rdkit.shape[0] != edge_attr_cm.shape[0]:
            logger.error(f"Mismatch in number of RDKit edges ({edge_attr_rdkit.shape[0]}) and CM edges ({edge_attr_cm.shape[0]}) for molecule {original_idx}. This indicates a serious issue.")
            raise QM7EdgeFeatureMismatchError(original_idx, edge_attr_rdkit.shape[0], edge_attr_cm.shape[0], message="Mismatch in RDKit and Coulomb Matrix edge features.")
        edge_attr: torch.Tensor = torch.cat([edge_attr_rdkit, edge_attr_cm], dim=-1)
    except QM7EdgeFeatureMismatchError:
        raise
    except Exception as e:
        logger.error(f"Error combining RDKit and Coulomb matrix edge features for molecule {original_idx}: {e}")
        raise QM7GraphConstructionError(original_idx, message=f"Error combining RDKit and CM edge features.", original_exception=e) from e

    try:
        y: torch.Tensor = torch.tensor([target_energy], dtype=torch.float).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error converting target energy {target_energy} to tensor for molecule {original_idx}: {e}")
        raise QM7GraphConstructionError(original_idx, message=f"Error with target energy tensor conversion.", original_exception=e) from e

    try:
        data: Data = Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if data.num_nodes == 0 and data.x.size(0) == 0:
            logger.warning(f"Created a PyG Data object for molecule {original_idx} with 0 nodes. This might indicate an issue with the molecule or previous processing steps.")

        data.num_nodes = torch.tensor(data.x.size(0), dtype=torch.long)
        data.idx = torch.tensor(original_idx, dtype=torch.long)
    except Exception as e:
        logger.error(f"Error instantiating PyG Data object for molecule {original_idx}: {e}")
        raise QM7GraphConstructionError(original_idx, message=f"Failed to create PyG Data object.", original_exception=e) from e

    return data

def filter_by_max_nodes(data: Data, max_nodes: int) -> bool:
    """
    A pre-filter function to exclude PyTorch Geometric `Data` objects
    (graphs) that have more nodes (atoms) than a specified maximum limit.

    This function can be used in conjunction with `functools.partial` to
    create a specialized filter for a specific `max_nodes` value, for example:
    `partial(filter_by_max_nodes, max_nodes=15)`.

    Args:
        data (Data): The PyTorch Geometric `Data` object to evaluate.
        max_nodes (int): The maximum allowed number of nodes (atoms) in the graph.

    Returns:
        bool: `True` if the graph's number of nodes is less than or equal to
              `max_nodes`, indicating it passes the filter. `False` otherwise.
    """
    return data.x.size(0) <= max_nodes


def filter_by_min_nodes(data: Data, min_nodes: int) -> bool:
    """
    A pre-filter function to exclude PyTorch Geometric `Data` objects
    (graphs) that have fewer nodes (atoms) than a specified minimum limit.

    This function can be used similarly to `filter_by_max_nodes` with
    `functools.partial`.

    Args:
        data (Data): The PyTorch Geometric `Data` object to evaluate.
        min_nodes (int): The minimum allowed number of nodes (atoms) in the graph.

    Returns:
        bool: `True` if the graph's number of nodes is greater than or equal to
              `min_nodes`, indicating it passes the filter. `False` otherwise.
    """
    return data.x.size(0) >= min_nodes

def filter_by_num_carbons(data: Data, min_carbons: int) -> bool:
    """
    A pre-filter function to exclude PyTorch Geometric `Data` objects
    (graphs) that have fewer than a specified minimum number of carbon atoms.

    It assumes that the atomic numbers are stored in the `data.z` attribute
    of the PyG `Data` object and that the atomic number for Carbon is 6.

    This function can be used with `functools.partial` to define a filter
    for a specific minimum carbon count, for example:
    `partial(filter_by_num_carbons, min_carbons=1)`.

    Args:
        data (Data): The PyTorch Geometric `Data` object to evaluate.
        min_carbons (int): The minimum required number of carbon atoms in the graph.

    Returns:
        bool: `True` if the graph contains `min_carbons` or more carbon atoms,
              indicating it passes the filter. `False` otherwise.
    """
    num_carbons: int = (data.z == 6).sum().item()
    return num_carbons >= min_carbons

def process_qm7_data(
    sdf_file: Path,
    energies_file: Path,
    mat_file: Path,
    intermediate_chunk_output_dir: Path,
    chunk_size: int,
    pre_filter: Optional[Any] = None,
) -> Path:
    """
    Processes raw QM7 dataset files (SDF, CSV for energies, and .mat for Coulomb matrices/atomic charges)
    into a collection of PyTorch Geometric (PyG) `Data` objects. These `Data` objects are then
    saved in smaller, manageable chunks as `.pt` files within a specified temporary directory.

    This function performs the following key steps:
    1. **Loads raw data**: Reads molecular structures from an SDF file, atomization
        energies from a CSV, and Coulomb matrices/atomic charges from a .mat file.
    2. **Data Alignment and Consistency Check**:
        - Truncates the list of RDKit molecules if their count exceeds the number of
          entries in the MAT or energy files to ensure consistent data dimensions.
        - Identifies and excludes SDF molecules whose atomic compositions are
          inconsistent with the MAT file's atomic charge data, preventing
          downstream errors.
    3. **Graph Construction**: Iterates through the aligned data to construct
        individual PyG `Data` objects for each molecule. Each `Data` object
        includes:
        - **Node features (`x`)**: One-hot encoded atom type, atomic number,
          aromaticity, hybridization (SP, SP2, SP3), number of hydrogens,
          atomic charge (from .mat file), and diagonal element of the Coulomb matrix.
        - **Atomic numbers (`z`)**: Atomic numbers of atoms in the molecule.
        - **3D Positions (`pos`)**: Atomic coordinates from the SDF file.
        - **Edge index (`edge_index`)**: Connectivity of bonded atoms.
        - **Edge attributes (`edge_attr`)**: One-hot encoded RDKit bond types
          (single, double, triple, aromatic) combined with the off-diagonal
          elements of the Coulomb matrix for bonded atoms.
        - **Target property (`y`)**: Atomization energy for the molecule.
        - **Original Index (`idx`)**: The original index of the molecule in the dataset.
    4. **Pre-filtering**: Applies an optional `pre_filter` (if provided) to each
        newly created `Data` object. This allows for filtering out graphs that
        do not meet certain criteria (e.g., maximum/minimum number of nodes)
        *before* saving.
    5. **Chunked Saving**: Organizes the generated PyG `Data` objects into
        chunks and saves each chunk as a separate `.pt` file in the
        `intermediate_chunk_output_dir`. This approach is memory-efficient
        and facilitates parallel processing or resumed operations.
    6. **Error Handling & Logging**: Provides robust error handling for data
        loading, graph construction, filtering, and saving, logging detailed
        information about any issues encountered and raising specific custom
        exceptions for critical failures.

    Note: This function only performs initial processing and pre-filtering.
    It does *not* apply any PyG `pre_transforms` (e.g., normalization, feature
    engineering) or merge the saved chunks into a single dataset file. Those
    steps are typically handled in subsequent stages of the data pipeline.

    Args:
        sdf_file (Path): Path to the SDF file containing molecular structures.
        energies_file (Path): Path to the CSV file containing atomization energies.
        mat_file (Path): Path to the .mat file containing Coulomb matrices and
                         atomic charges.
        intermediate_chunk_output_dir (Path): Directory where processed PyG
                                               data chunks will be saved. This
                                               directory will be cleared before
                                               processing if it exists.
        chunk_size (int): The maximum number of PyG `Data` objects to save in
                          each individual chunk file.
        pre_filter (Optional[Any]): An optional pre-filter object. If provided,
                                     it is expected to have a `transforms`
                                     attribute which is an iterable of callables.
                                     Each callable in `transforms` should accept
                                     a PyG `Data` object and return `True` if the
                                     data passes the filter, `False` otherwise.
                                     Graphs failing any filter will be skipped.

    Returns:
        Path: The path to the directory where the processed PyG data chunks are saved.

    Raises:
        QM7DataLoadError: If any of the input data files (SDF, CSV, .mat) cannot
                          be found, are empty, or contain malformed data.
        QM7DataAlignmentError: If there's a critical mismatch in the number of
                               entries across the input files, leading to no
                               effective data for processing.
        FileSystemError: If the intermediate output directory cannot be created
                         or cleared.
        QM7GraphConstructionError: If there's a general error during graph
                                   construction (e.g., missing conformer, general
                                   graph construction failure).
        QM7InvalidMoleculeError: If an RDKit molecule is None or has no atoms.
        QM7FeatureExtractionError: If there's an error extracting specific features
                                   (e.g., node features, bond features, Coulomb
                                   matrix features) for a molecule.
        QM7EdgeFeatureMismatchError: If the number of RDKit-derived edges and
                                     Coulomb matrix-derived edges do not match,
                                     indicating a serious internal inconsistency.
        QM7PreFilterError: If an error occurs during the application of a pre-filter.
        QM7ChunkSavingError: If an error occurs while saving a processed data chunk.
        QM7SpecificProcessingError: A general error indicating that no data chunks
                                    were successfully saved, suggesting a major issue
                                    with the processing pipeline.
    """
    logger.info(f"Starting data processing for SDF: {sdf_file}, Energies: {energies_file}, MAT: {mat_file}")

    try:
        molecules: List[Chem.Mol] = load_sdf_molecules(sdf_file)
        energies: np.ndarray = load_atomization_energies_pandas(energies_file)
        coulomb_matrices, atomic_charges = load_qm7_mat_data(mat_file)
    except QM7DataLoadError as e:
        logger.critical(f"Critical data loading error: {e}. Processing aborted.")
        raise

    if not molecules:
        logger.error("SDF molecules failed to load or were empty. Processing aborted.")
        raise QM7DataLoadError(sdf_file, "SDF", message="No molecules loaded from SDF file.")
    if energies is None or energies.size == 0:
        logger.error("Atomization energies failed to load or were empty. Processing aborted.")
        raise QM7DataLoadError(energies_file, "CSV", message="No atomization energies loaded from CSV file.")
    if coulomb_matrices is None or atomic_charges is None or coulomb_matrices.size == 0 or atomic_charges.size == 0:
        logger.error("Coulomb matrices or atomic charges failed to load or were empty. Processing aborted.")
        raise QM7DataLoadError(mat_file, "MAT", message="No Coulomb matrices or atomic charges loaded from MAT file.")

    num_mat_entries: int = coulomb_matrices.shape[0]
    num_energy_entries: int = len(energies)

    effective_data_size: int = min(len(molecules), num_mat_entries, num_energy_entries)

    if effective_data_size < len(molecules):
        logger.warning(f"SDF file has {len(molecules)} entries, but MAT/Energy files "
                                     f"only have {effective_data_size}. Truncating molecules list "
                                     f"to {effective_data_size} to avoid index out of bounds errors.")
        molecules = molecules[:effective_data_size]

    if effective_data_size == 0:
        logger.error("No effective data entries to process after initial loading and size alignment. Aborting.")
        raise QM7DataAlignmentError(message="No effective data entries to process after initial loading and size alignment.")

    sdf_indices_to_exclude: Set[int] = identify_sdf_indices_to_exclude(molecules, atomic_charges)

    try:
        shutil.rmtree(intermediate_chunk_output_dir, ignore_errors=True)
        intermediate_chunk_output_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Created temporary directory for intermediate chunks: {intermediate_chunk_output_dir}")
    except OSError as e:
        logger.critical(f"Failed to create or clean intermediate chunk directory {intermediate_chunk_output_dir}: {e}. Processing aborted.")
        raise FileSystemError(intermediate_chunk_output_dir, message=f"Failed to create/clean intermediate directory.", original_exception=e) from e

    total_molecules_to_process: int = len(molecules)
    processed_count: int = 0
    skipped_count: int = 0
    filtered_count: int = 0

    saved_chunk_files: List[Path] = []

    filters_to_apply: List[Callable[[Data], bool]] = pre_filter.transforms if pre_filter and hasattr(pre_filter, 'transforms') else []

    num_chunks: int = (total_molecules_to_process + chunk_size - 1) // chunk_size
    if total_molecules_to_process == 0:
        num_chunks = 0

    logger.info(f"Starting chunked graph creation with chunk size: {chunk_size}. Total molecules to process: {total_molecules_to_process}. Expected chunks: {num_chunks}")

    for chunk_num, start_idx in enumerate(tqdm(range(0, total_molecules_to_process, chunk_size), desc="Creating PyG Chunks"), 1):
        end_idx: int = min(start_idx + chunk_size, total_molecules_to_process)
        current_chunk_data_list: List[Data] = []

        for i in range(start_idx, end_idx):
            mol: Chem.Mol = molecules[i]
            original_idx: int = i

            if original_idx in sdf_indices_to_exclude:
                logger.debug(f"Skipping molecule {original_idx} due to consistency exclusion.")
                skipped_count += 1
                continue

            try:
                data: Data = create_pyg_graph(mol, atomic_charges[i], coulomb_matrices[i], energies[i], original_idx)

                filter_passed: bool = True
                for _filter_func in filters_to_apply:
                    try:
                        if not _filter_func(data):
                            filter_passed = False
                            logger.debug(f"Molecule {original_idx} filtered by {_filter_func.__name__ if hasattr(_filter_func, '__name__') else 'unknown filter'}.")
                            break
                    except Exception as filter_e:
                        logger.error(f"Error applying filter '{_filter_func.__name__ if hasattr(_filter_func, '__name__') else 'unknown filter'}' to molecule {original_idx}: {filter_e}. Molecule will be skipped.")
                        raise QM7PreFilterError(original_idx, _filter_func.__name__ if hasattr(_filter_func, '__name__') else 'unknown_filter', original_exception=filter_e) from filter_e

                if not filter_passed:
                    filtered_count += 1
                    continue

                current_chunk_data_list.append(data)
                processed_count += 1
            except (QM7GraphConstructionError, QM7InvalidMoleculeError, QM7FeatureExtractionError,
                    QM7EdgeFeatureMismatchError, QM7PreFilterError) as e:
                logger.error(f"Specific error processing molecule {original_idx} in chunk {chunk_num}: {e}. Skipping this molecule.")
                skipped_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing molecule {original_idx} in chunk {chunk_num}: {e}. Skipping this molecule.")
                skipped_count += 1

        if current_chunk_data_list:
            chunk_filename: Path = intermediate_chunk_output_dir / f"chunk_{chunk_num:02d}_{start_idx}-{end_idx-1}.pt"
            try:
                torch.save(current_chunk_data_list, chunk_filename)
                saved_chunk_files.append(chunk_filename)
                logger.info(f"Saved chunk {chunk_num} ({len(current_chunk_data_list)} graphs) to {chunk_filename}")
            except Exception as e:
                logger.error(f"Error saving chunk {chunk_num} to {chunk_filename}: {e}. This chunk's data might be lost.")
                raise QM7ChunkSavingError(chunk_filename, message=f"Failed to save chunk {chunk_num}", original_exception=e) from e
        else:
            logger.warning(f"--- Finalizing Chunk {chunk_num:02d}. No valid graphs generated or saved for this chunk. ---")

    logger.info(f"Initial PyG graph creation complete. Raw chunks saved to: {intermediate_chunk_output_dir}")
    logger.info(f"Total graphs created and saved to chunks: {processed_count}")
    logger.info(f"Total molecules skipped (due to consistency issues, processing errors, or chunk saving errors): {skipped_count}")
    logger.info(f"Total graphs filtered by pre_filter: {filtered_count}")

    total_accounted: int = processed_count + skipped_count + filtered_count
    if total_accounted != total_molecules_to_process:
        logger.warning(f"Discrepancy in pyg_qm7_processing: Total created graphs ({processed_count}) + skipped ({skipped_count}) "
                                     f"+ filtered ({filtered_count}) does not equal initial effective molecules ({total_molecules_to_process}). "
                                     f"Total accounted: {total_accounted}. This might indicate an unexpected issue during processing or counting.")
    else:
        logger.info("All molecules accounted for by pyg_qm7_processing (created, skipped, or filtered).")

    if not saved_chunk_files:
        logger.error(f"No chunks were successfully saved to {intermediate_chunk_output_dir}. This indicates a major issue.")
        raise QM7SpecificProcessingError(message=f"No data chunks were successfully saved to {intermediate_chunk_output_dir} after processing.")

    return intermediate_chunk_output_dir
