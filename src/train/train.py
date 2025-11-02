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
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool



def load_data(xyz_path: str) -> List[ase.Atoms]:
    """Loads atom configurations, checking for required fields."""
    logging.info(f"Loading data from '{xyz_path}'...")
    try:
        atoms_list = ase.io.read(xyz_path, index=":")
        logging.info(f"Successfully loaded {len(atoms_list)} configurations.")
        # Check for the keys needed for training and evaluation
        if 'energy_mace_off' not in atoms_list[0].info or 'total_delta_energy' not in atoms_list[0].info:
             raise KeyError("Dataset is missing 'energy_mace_off' or 'total_delta_energy' in the info field.")
        return atoms_list
    except Exception as e:
        logging.error(f"Failed to load or parse dataset: {e}")
        raise

class AtomsDataset(torch.utils.data.Dataset):
    def __init__(self, atoms_list: List[ase.Atoms], r_max: float, z_map: Dict[int, int]):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.z_map = z_map
        self.num_classes = len(z_map)

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        edge_src, edge_dst, shifts_images = neighbor_list("ijS", a=atoms, cutoff=self.r_max, self_interaction=False)
        
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        atomic_numbers_np = atoms.get_atomic_numbers()
        atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)
        
        try:
            node_indices = torch.tensor([self.z_map[z] for z in atomic_numbers_np], dtype=torch.long)
            node_attrs = F.one_hot(node_indices, num_classes=self.num_classes).to(torch.float64)
        except KeyError as e:
            logging.error(f"Atomic number {e} not in model's z_map.")
            raise
        
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
        shifts = torch.tensor(shifts_images, dtype=torch.float64) @ cell

        target_delta_energy = atoms.info['total_delta_energy']
        true_final_energy = atoms.info['energy_mace_off']
        
        data = Data(
            pos=positions,
            atomic_numbers=atomic_numbers,
            node_attrs=node_attrs,
            edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            shifts=shifts,
            cell=cell.unsqueeze(0),
            pbc=pbc,
            y=torch.tensor([target_delta_energy], dtype=torch.float64), # y is now the delta
            true_final_energy=torch.tensor([true_final_energy], dtype=torch.float64) # Store final energy for metrics
        )
        return data

def pyg_collate(batch):
    return Batch.from_data_list(batch)

class DeltaEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: Dict[str, torch.Tensor], ref: Data) -> torch.Tensor:
        return self.mse(pred["delta_energy"], ref.y)

def evaluate(model: nn.Module, data_loader: PyGDataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float, float, float]:
    """
    Evaluates the model and computes loss, delta MAE, delta RMSE, and final energy MAE per atom.
    """
    model.eval()
    total_loss = 0.0
    all_pred_deltas = []
    all_true_deltas = []
    
    # Store absolute errors per atom for the final energy metric
    all_final_energy_abs_errors_per_atom = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')

            output = model(data_dict)
            loss = loss_fn(output, batch)
            total_loss += loss.item() * batch.num_graphs

            all_pred_deltas.append(output['delta_energy'].cpu())
            all_true_deltas.append(batch.y.cpu())
            
            # --- MAE per atom calculation ---
            pred_final_energies = output['energy'].cpu()
            true_final_energies = batch.true_final_energy.cpu()
            
            # Get the number of atoms for each graph in the batch
            # `ptr` gives the cumulative sum of atoms, so the difference gives the count for each graph
            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).cpu().to(pred_final_energies.dtype)
            
            # Calculate absolute error per atom for each graph and append
            abs_error_per_atom = torch.abs(pred_final_energies - true_final_energies) / num_atoms
            all_final_energy_abs_errors_per_atom.append(abs_error_per_atom)

    avg_loss = total_loss / len(data_loader.dataset)
    
    all_pred_deltas = torch.cat(all_pred_deltas)
    all_true_deltas = torch.cat(all_true_deltas)
    
    # Concatenate all collected per-atom absolute errors
    all_final_energy_abs_errors_per_atom = torch.cat(all_final_energy_abs_errors_per_atom)
    
    # Metrics on the delta (this is what the model is trained on)
    delta_mae = torch.mean(torch.abs(all_pred_deltas - all_true_deltas)).item()
    delta_rmse = torch.sqrt(torch.mean((all_pred_deltas - all_true_deltas)**2)).item()
    
    # New metric: MAE on the final total energy per atom
    final_energy_mae_per_atom = torch.mean(all_final_energy_abs_errors_per_atom).item()
    
    print(f"Final Energy MAE per atom (metric): {final_energy_mae_per_atom:.4f} eV/atom")
    
    return avg_loss, delta_mae, delta_rmse, final_energy_mae_per_atom


def train_model(
    model: nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
):
    """Main training loop."""
    best_val_mae = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Convert batch to dictionary for model input
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')

            output = model(data_dict)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # --- Evaluation ---
        val_loss, val_delta_mae, val_delta_rmse, val_final_mae_per_atom = evaluate(
            model, val_loader, loss_fn, device
        )
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Delta MAE: {val_delta_mae:.4f} | "
            f"Val Final Energy MAE/atom: {val_final_mae_per_atom:.4f} eV/atom"
        )

        if scheduler:
            scheduler.step(val_final_mae_per_atom)

        if val_final_mae_per_atom < best_val_mae:
            best_val_mae = val_final_mae_per_atom
            print(f"New best validation MAE: {best_val_mae:.4f}. Saving model...")
            # torch.save(model.state_dict(), "best_model.pt") # Example saving

    print("Training complete.")


# --- NEW HELPER FUNCTION 1 ---
def precompute_dataset(
    model: 'DualReadoutMACE', dataloader: PyGDataLoader, device: torch.device
) -> List[Data]:
    """
    Runs the expensive base model once and stores the features, base energy,
    and labels for fast training.
    """
    logging.info(f"Starting pre-computation for {len(dataloader.dataset)} samples...")
    model.eval()
    precomputed_data_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            data_dict = batch.to_dict()
            if "pos" in data_dict:
                data_dict["positions"] = data_dict.pop("pos")

            # Run the expensive part
            base_output = model.mace_model(data_dict, compute_force=False)
            features = model.features.cpu()  # [N_atoms_in_batch, 128]
            base_energies = base_output["energy"].cpu()  # [N_graphs_in_batch]
            model.features = None  # Reset hook

            # De-collate the batch to get individual graph data
            original_data_list = batch.to_data_list()
            ptr = batch.ptr.cpu()

            for i in range(len(original_data_list)):
                original_data = original_data_list[i]
                start, end = ptr[i], ptr[i + 1]

                # Extract features for this specific graph
                graph_features = features[start:end]
                graph_base_energy = base_energies[i]  # Get the energy for this graph

                # Create a new, lightweight Data object
                new_data = Data(
                    x=graph_features,  # The pre-computed features
                    y=original_data.y.cpu(),
                    base_energy=graph_base_energy,  # Store the base energy
                    true_final_energy=original_data.true_final_energy.cpu(),
                    #
                    # --- FIX IS HERE ---
                    # Use dictionary access to get the tensor, not the method
                    node_attrs=original_data["node_attrs"].cpu(),
                    #
                    # This is a placeholder; the new DataLoader will create the correct batch map
                    batch_map=torch.zeros(graph_features.shape[0], dtype=torch.long),
                )
                precomputed_data_list.append(new_data)

    logging.info("Pre-computation complete.")
    return precomputed_data_list


# --- NEW HELPER FUNCTION 2 ---
def evaluate_fast(
    model: 'DualReadoutMACE',
    data_loader: PyGDataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluates the model using PRE-COMPUTED features.
    This is a modified version of the `evaluate` function from train.py.
    """
    model.eval()
    total_loss = 0.0
    all_pred_deltas = []
    all_true_deltas = []
    all_final_energy_abs_errors_per_atom = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            # === Manually replicate the model's forward pass ===
            processed_features = batch.x  # 'x' is where we stored features
            if model.use_pca:
                processed_features = model.pca_layer(processed_features)

            atomic_delta_prediction = model.delta_readout(processed_features)

            if isinstance(model.atomic_energy_shifts, torch.Tensor):
                #
                # --- FIX IS HERE ---
                # Use dictionary access to get the tensor, not the method
                node_indices = torch.argmax(batch["node_attrs"], dim=1)
                #
                shifts = model.atomic_energy_shifts[node_indices].unsqueeze(-1)
                total_atomic_delta = atomic_delta_prediction + shifts
            else:
                total_atomic_delta = atomic_delta_prediction

            batch_map = (
                batch.batch
            )  # This is the *new* batch map from val_loader_fast
            delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)

            base_energy = batch.base_energy
            final_energy = base_energy + delta_energy
            # ===================================================

            output = {"energy": final_energy, "delta_energy": delta_energy}

            # --- Now, the original evaluation logic ---
            loss = loss_fn(
                output, batch
            )  # loss_fn compares output["delta_energy"] and batch.y
            total_loss += loss.item() * batch.num_graphs

            all_pred_deltas.append(output["delta_energy"].cpu())
            all_true_deltas.append(batch.y.cpu())

            pred_final_energies = output["energy"].cpu()
            true_final_energies = batch.true_final_energy.cpu()

            num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).cpu().to(pred_final_energies.dtype)

            abs_error_per_atom = (
                torch.abs(pred_final_energies - true_final_energies) / num_atoms
            )
            all_final_energy_abs_errors_per_atom.append(abs_error_per_atom)

    avg_loss = total_loss / len(data_loader.dataset)
    all_pred_deltas = torch.cat(all_pred_deltas)
    all_true_deltas = torch.cat(all_true_deltas)
    all_final_energy_abs_errors_per_atom = torch.cat(
        all_final_energy_abs_errors_per_atom
    )

    delta_mae = torch.mean(torch.abs(all_pred_deltas - all_true_deltas)).item()
    delta_rmse = torch.sqrt(
        torch.mean((all_pred_deltas - all_true_deltas) ** 2)
    ).item()
    final_energy_mae_per_atom = torch.mean(
        all_final_energy_abs_errors_per_atom
    ).item()

    return avg_loss, delta_mae, delta_rmse, final_energy_mae_per_atom

