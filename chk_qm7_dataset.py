# chk_qm7_dataset.py
import torch
from pathlib import Path
import logging
import torch_geometric.data.data
import torch_geometric.data.storage


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dataset(dataset_path: Path):
    """ 
    Performs basic checks on a saved PyTorch Geometric dataset to verify its integrity.
    Checks for the presence of essential attributes and their data types.
    """
    logger.info(f"Checking dataset at: {dataset_path}")

    try:
        try:
            safe_classes = [
                torch_geometric.data.data.Data,
                torch_geometric.data.data.DataEdgeAttr,
                torch_geometric.data.data.DataTensorAttr,
                torch_geometric.data.storage.GlobalStorage,
            ]
            torch.serialization.add_safe_globals(safe_classes)
            logger.info("Added torch_geometric classes to PyTorch's safe globals for loading.")
        except AttributeError:
            logger.warning("torch.serialization.add_safe_globals not available. "
                           "If using PyTorch 2.6+, ensure compatibility or manually handle weights_only.")

        data_list = torch.load(str(dataset_path))
        logger.info(f"Successfully loaded {len(data_list)} graphs from {dataset_path}")

        if not isinstance(data_list, list):
            logger.error(f"Expected a list of Data objects, but got: {type(data_list)}")
            return

        if not data_list:
            logger.warning("The dataset is empty (contains no graphs).")
            return

        """ Check the first graph as a representative sample """ 
        sample_data = data_list[0]

        required_attributes = ['x', 'edge_index', 'edge_attr', 'y', 'z', 'pos', 'idx']
        expected_additional_attributes = ['num_nodes'] 

        logger.info("\n--- Checking first graph attributes ---")
        for attr_name in required_attributes:
            if not hasattr(sample_data, attr_name):
                logger.error(f"  Missing required attribute: {attr_name}")
                return # Critical error, stop check
            attr_value = getattr(sample_data, attr_name)
            shape_info = attr_value.shape if hasattr(attr_value, 'shape') else 'N/A'
            dtype_info = attr_value.dtype if hasattr(attr_value, 'dtype') else type(attr_value).__name__
            logger.info(f"  Attribute '{attr_name}' exists | Shape: {shape_info} | Dtype: {dtype_info}")
        
        for attr_name in expected_additional_attributes:
            if hasattr(sample_data, attr_name):
                attr_value = getattr(sample_data, attr_name)
                shape_info = attr_value.shape if hasattr(attr_value, 'shape') else 'N/A'
                dtype_info = attr_value.dtype if hasattr(attr_value, 'dtype') else type(attr_value).__name__
                logger.info(f"  Attribute '{attr_name}' exists (expected additional) | Shape: {shape_info} | Dtype: {dtype_info}")
            else:
                logger.warning(f"  Expected additional attribute '{attr_name}' not found.")


        logger.info("\n--- Performing specific data type and tensor type checks ---")
        if isinstance(sample_data.edge_index, torch.Tensor) and sample_data.edge_index.dtype != torch.long:
            logger.error("  edge_index should have dtype torch.long")
        
        if isinstance(sample_data.y, torch.Tensor) and sample_data.y.dtype != torch.float:
            logger.error("  y (target) should have dtype torch.float")
        
        if hasattr(sample_data, 'num_nodes'):
            if not isinstance(sample_data.num_nodes, torch.Tensor):
                logger.error("  num_nodes is NOT a torch.Tensor. It MUST be a torch.LongTensor.")
            elif sample_data.num_nodes.dtype != torch.long:
                logger.error("  num_nodes should have dtype torch.long.")
            else:
                logger.info("  num_nodes is correctly a torch.LongTensor.")

        if hasattr(sample_data, 'idx'):
            if not isinstance(sample_data.idx, torch.Tensor):
                logger.error("  idx is NOT a torch.Tensor. It MUST be a torch.LongTensor.")
            elif sample_data.idx.dtype != torch.long:
                logger.error("  idx should have dtype torch.long.")
            else:
                logger.info("  idx is correctly a torch.LongTensor.")
        
        logger.info("\n--- Consistency checks across a few graphs ---")
    
        x_dim = sample_data.x.shape[1] if sample_data.x.ndim > 1 else 'N/A'
        edge_attr_dim = sample_data.edge_attr.shape[1] if sample_data.edge_attr.ndim > 1 else 'N/A'

        num_checks = min(5, len(data_list))
        for i in range(num_checks):
            current_data = data_list[i]
            if current_data.x.ndim > 1 and current_data.x.shape[1] != x_dim:
                logger.warning(f"  x feature dimension mismatch in graph {i}: Expected {x_dim}, Got {current_data.x.shape[1]}")
            if current_data.edge_attr.ndim > 1 and current_data.edge_attr.shape[1] != edge_attr_dim:
                logger.warning(f"  edge_attr feature dimension mismatch in graph {i}: Expected {edge_attr_dim}, Got {current_data.edge_attr.shape[1]}")
            if current_data.y.numel() != 1:
                logger.warning(f"  y (target) in graph {i} does not have exactly one element: {current_data.y.numel()}")


        logger.info("\nBasic dataset checks completed. Review logged messages for any errors or warnings.")

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
    except Exception as e:
        logger.error(f"An error occurred while checking the dataset: {e}")


if __name__ == "__main__":
    dataset_file = Path(r"/home/shahram/Chem_Data/qm7/processed/qm7_graph_data_curated.pt")
    check_dataset(dataset_file)
