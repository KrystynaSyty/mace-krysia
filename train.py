import argparse
import logging
import os
import random
from typing import Dict, List, Tuple

import ase.io
from ase.neighborlist import neighbor_list
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import radius_graph

# Assuming 'models.py' is in the same directory or accessible in the python path
from models import DualReadoutMACE

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class AtomsDataset(torch.utils.data.Dataset):
    """Custom Dataset for ASE Atoms objects."""
    def __init__(self, atoms_list: List[ase.Atoms], r_max: float, z_map: Dict[int, int]):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.z_map = z_map
        self.num_classes = len(z_map)

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        # Get neighbors using the cutoff radius
        edge_src, edge_dst, shifts_images = neighbor_list(
            "ijS", a=atoms, cutoff=self.r_max, self_interaction=False
        )

        # Extract necessary information and convert to tensors
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        atomic_numbers_np = atoms.get_atomic_numbers()
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)
        
        # Create node_attrs by mapping atomic numbers to indices and then one-hot encoding
        try:
            node_indices = torch.tensor([self.z_map[z] for z in atomic_numbers_np], dtype=torch.long)
            node_attrs = F.one_hot(node_indices, num_classes=self.num_classes).to(torch.float64)
        except KeyError as e:
            logging.error(f"Atomic number {e} not found in the base model's atomic number list. Please check your dataset or base model.")
            raise
        
        # Convert cell to numpy array before tensor to avoid warning
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
        
        # Calculate shifts in cartesian coordinates
        shifts = torch.tensor(shifts_images, dtype=torch.float64) @ cell
        
        # The target for our loss function is the corrected total energy
        # E_true = E_mace_off + delta_energy
        # We assume E_mace_off is the energy from the base MACE model without the delta correction.
        target_energy = atoms.info['energy_mace_off'] + atoms.info['delta_energy']
        
        data = Data(
            pos=positions,
            atomic_numbers=atomic_numbers,
            node_attrs=node_attrs,
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            shifts=shifts,
            cell=cell.unsqueeze(0), # MACE expects a batch dimension for cell
            pbc=pbc,
            y=torch.tensor([target_energy], dtype=torch.float64),
            delta_energy=torch.tensor([atoms.info['delta_energy']], dtype=torch.float64)
        )
        return data

def pyg_collate(batch):
    """Collate function for PyTorch Geometric DataLoader."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

def load_data(xyz_path: str) -> List[ase.Atoms]:
    """Loads atom configurations from an extended XYZ file."""
    logging.info(f"Loading data from '{xyz_path}'...")
    try:
        atoms_list = ase.io.read(xyz_path, index=":")
        logging.info(f"Successfully loaded {len(atoms_list)} configurations.")
        # Check if the required info fields are present
        if 'delta_energy' not in atoms_list[0].info or 'energy_mace_off' not in atoms_list[0].info:
             raise KeyError("Dataset is missing 'delta_energy' or 'energy_mace_off' in the info field.")
        return atoms_list
    except Exception as e:
        logging.error(f"Failed to load or parse dataset: {e}")
        raise

class DeltaEnergyLoss(nn.Module):
    """
    Custom loss function for the delta learning task.
    Calculates the Mean Squared Error on the total energy.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: Dict[str, torch.Tensor], ref: Data) -> torch.Tensor:
        """
        Calculates loss.
        pred: Dictionary from model forward pass, expecting 'energy' key.
        ref: PyG Data or Batch object with 'y' as the target total energy.
        """
        return self.mse(pred["energy"], ref.y)

def evaluate(model: nn.Module, data_loader: PyGDataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate the model on the validation set.
    Returns average loss, MAE on delta_energy, and RMSE on delta_energy.
    """
    model.eval()
    total_loss = 0.0
    all_pred_deltas = []
    all_true_deltas = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')

            # The model's forward pass gives the final, corrected energy
            output = model(data_dict)
            
            # Loss is calculated on the total energy
            loss = loss_fn(output, batch)
            total_loss += loss.item() * batch.num_graphs

            # To evaluate the delta learning, we explicitly calculate the predicted delta
            base_output = model.mace_model(data_dict, compute_force=False)
            predicted_delta = output['energy'] - base_output['energy']
            
            all_pred_deltas.append(predicted_delta.cpu())
            all_true_deltas.append(batch.delta_energy.cpu())

    avg_loss = total_loss / len(data_loader.dataset)
    
    all_pred_deltas = torch.cat(all_pred_deltas)
    all_true_deltas = torch.cat(all_true_deltas)
    
    mae = torch.mean(torch.abs(all_pred_deltas - all_true_deltas)).item()
    rmse = torch.sqrt(torch.mean((all_pred_deltas - all_true_deltas)**2)).item()
    
    return avg_loss, mae, rmse



