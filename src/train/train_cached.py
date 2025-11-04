# train.py
import logging
from typing import Dict, List, Tuple

import ase.io
from ase.neighborlist import neighbor_list
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader as StandardDataLoader, TensorDataset, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from models.models import DualReadoutMACE
from models.models_cached import CachedReadoutModel
from tqdm import tqdm

def evaluate_cached(
    model: CachedReadoutModel, 
    data_loader: StandardDataLoader, 
    loss_fn: "DeltaEnergyLoss", 
    device: torch.device
) -> Tuple[float, float, float, float]:
    """
    Evaluates the CachedReadoutModel.
    """
    model.eval()
    total_loss = 0.0
    all_pred_deltas = []
    all_true_deltas = []
    all_final_energy_abs_errors_per_atom = []

    with torch.no_grad():
        for batch in data_loader:
            # Move all tensors in the batch dictionary to the device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model(batch)
            
            # Loss function expects dicts: pred['delta_energy'] and ref['y']
            loss = loss_fn(output, batch)
            total_loss += loss.item() * batch['num_graphs']

            all_pred_deltas.append(output['delta_energy'].cpu())
            all_true_deltas.append(batch['y'].cpu())
            
            # --- MAE per atom calculation ---
            pred_final_energies = output['energy'].cpu()
            true_final_energies = batch['true_final_energy'].cpu()
            num_atoms = batch['num_atoms'].cpu().to(pred_final_energies.dtype)
            
            abs_error_per_atom = torch.abs(pred_final_energies - true_final_energies) / num_atoms
            all_final_energy_abs_errors_per_atom.append(abs_error_per_atom)

    avg_loss = total_loss / len(data_loader.dataset)
    
    all_pred_deltas = torch.cat(all_pred_deltas)
    all_true_deltas = torch.cat(all_true_deltas)
    all_final_energy_abs_errors_per_atom = torch.cat(all_final_energy_abs_errors_per_atom)
    
    delta_mae = torch.mean(torch.abs(all_pred_deltas - all_true_deltas)).item()
    delta_rmse = torch.sqrt(torch.mean((all_pred_deltas - all_true_deltas)**2)).item()
    final_energy_mae_per_atom = torch.mean(all_final_energy_abs_errors_per_atom).item()
    
    return avg_loss, delta_mae, delta_rmse, final_energy_mae_per_atom


def precompute_features_in_memory(
    feature_extractor: DualReadoutMACE,
    data_loader: PyGDataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs the dataset through the frozen base model ONCE and returns all
    features and targets as large tensors.
    """
    feature_extractor.eval()
    
    all_features = []
    all_targets = []
    all_base_energies = []
    all_true_final_energies = []
    all_num_atoms = []
    all_node_attrs = [] # Need this for the energy shifts

    logging.info(f"Starting in-memory pre-computation for {len(data_loader.dataset)} structures...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Pre-computing features"):
            batch = batch.to(device)
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')

            # Run the base MACE model, explicitly disabling force calculation
            output = feature_extractor.mace_model(data_dict, compute_force=False)
            all_base_energies.append(output['energy'].cpu())

            if feature_extractor.features is not None:
                num_atoms_per_graph = (batch.ptr[1:] - batch.ptr[:-1])
                
                all_features.append(feature_extractor.features.cpu())
                all_node_attrs.append(data_dict['node_attrs'].cpu())                
                # Store per-graph information
                all_targets.append(batch.y.cpu())
                all_true_final_energies.append(batch.true_final_energy.cpu())
                all_num_atoms.append(num_atoms_per_graph.cpu())
                                
                feature_extractor.features = None # Reset hook
            else:
                logging.warning("Hook did not capture features for a batch.")

    logging.info("Concatenating all data into large tensors...")
    
    # Per-atom tensors
    features_tensor = torch.cat(all_features, dim=0)
    node_attrs_tensor = torch.cat(all_node_attrs, dim=0)
    
    # Per-graph tensors
    targets_tensor = torch.cat(all_targets, dim=0)
    base_energies_tensor = torch.cat(all_base_energies, dim=0)
    true_final_energies_tensor = torch.cat(all_true_final_energies, dim=0)
    num_atoms_tensor = torch.cat(all_num_atoms, dim=0)

    # Build the 'ptr' array (cumulative sum of atoms) for the *entire* dataset
    ptr_tensor = torch.cat([torch.tensor([0]), torch.cumsum(num_atoms_tensor, dim=0)])
    
    logging.info(f"Pre-computation complete. Total atoms: {features_tensor.shape[0]}")
    
    return (
        features_tensor, 
        node_attrs_tensor,
        targets_tensor, 
        base_energies_tensor, 
        true_final_energies_tensor, 
        ptr_tensor
    )

# --- NEW: A Dataset to serve the cached tensors ---

class FastCachedDataset(Dataset):
    """
    A simple PyTorch Dataset that serves slices of the giant pre-computed tensors.
    """
    def __init__(self, features, node_attrs, targets, base_energies, true_final_energies, ptr):
        self.features = features
        self.node_attrs = node_attrs
        self.targets = targets
        self.base_energies = base_energies
        self.true_final_energies = true_final_energies
        self.ptr = ptr
        
        self.num_graphs = len(targets)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        start = self.ptr[idx]
        end = self.ptr[idx+1]
        
        return {
            'x': self.features[start:end],
            'node_attrs': self.node_attrs[start:end],
            'y': self.targets[idx],
            'base_energy': self.base_energies[idx],
            'true_final_energy': self.true_final_energies[idx],
            'num_atoms': end - start
        }

# --- NEW: Collate function for the FastCachedDataset ---

def fast_collate_fn(batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates the dictionaries from FastCachedDataset into a single batch dictionary.
    This is necessary to create the 'batch_map'.
    """
    features = torch.cat([item['x'] for item in batch_list], dim=0)
    node_attrs = torch.cat([item['node_attrs'] for item in batch_list], dim=0)
    
    y = torch.stack([item['y'] for item in batch_list])
    base_energy = torch.stack([item['base_energy'] for item in batch_list])
    true_final_energy = torch.stack([item['true_final_energy'] for item in batch_list])
    num_atoms = torch.tensor([item['num_atoms'] for item in batch_list])

    # Create the batch_map (atom_index -> graph_index) for this mini-batch
    batch_map = torch.repeat_interleave(torch.arange(len(batch_list)), repeats=num_atoms)
    
    return {
        'x': features,
        'node_attrs': node_attrs,
        'y': y,
        'base_energy': base_energy,
        'true_final_energy': true_final_energy,
        'batch_map': batch_map,
        'num_atoms': num_atoms,
        'num_graphs': len(batch_list)
    }

