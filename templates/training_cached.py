# training_cached.py
import logging
import os
import time  # <-- Added
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as StandardDataLoader, TensorDataset, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool
from tqdm import tqdm
from ase import Atoms
from mace.calculators import MACECalculator


from models.models import DualReadoutMACE, compute_E_statistics_vectorized, PCALayer, MLPReadout 
from models.helping_func import get_vacuum_energies, print_model_summary
from models.models_cached import CachedReadoutModel
# We still need the *original* dataset to load the .xyz files
from train.train import DeltaEnergyLoss, load_data, pyg_collate, AtomsDataset 
from train.train_cached import fast_collate_fn, FastCachedDataset, precompute_features_in_memory, evaluate_cached

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    start_time = time.time() 
    #Configuration
    training_dataset_path = "{training_dataset.xyz}"
    validation_dataset_path = "{validation_dataset.xyz}"
    base_model_path = "{base_model.model}"
    high_accuracy_model_path = "{hoght_model.model}"
    epochs = "{N_epochs}"
    lr = "{lr}"
    batch_size = 10
    output_dir = "delta_model_checkpoints"
    
    mlp_hidden_features = "{n_hidden}" # "auto" or an np.array 
    mlp_activation = {"activation_type"} #silu or swiglu
    use_pca = {"True"} #bool True or False
    pca_variance_threshold = {"threshold"}
    #End of configuration
    os.makedirs(output_dir, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu" 
    device = torch.device(device_str)
    logging.info(f"Using device: {device_str}")

    # --- 1. Load Data ---
    logging.info(f"Loading training data from: '{training_dataset_path}'")
    training_atoms_list = load_data(training_dataset_path)
    logging.info(f"Loading validation data from: '{validation_dataset_path}'")
    validation_atoms_list = load_data(validation_dataset_path)
    full_atoms_list = training_atoms_list + validation_atoms_list
    logging.info(f"Total structures loaded: {len(full_atoms_list)}")

    # --- 2. Load Base Model and Calculate Shifts  ---
    base_mace_model = torch.load(base_model_path, map_location=device)
    base_mace_model.to(dtype=torch.float64).eval()
    
    r_max = base_mace_model.r_max.item()
    atomic_numbers_list = base_mace_model.atomic_numbers.tolist()
    z_map = {z: i for i, z in enumerate(atomic_numbers_list)}
    n_species = len(atomic_numbers_list)

    calc_mace_off = MACECalculator(model_paths=high_accuracy_model_path, device=device_str)
    calc_mace_mp = MACECalculator(model_paths=base_model_path, device=device_str)
    all_zs_in_dataset = [z for atoms in full_atoms_list for z in atoms.get_atomic_numbers()]
    vacuum_energies = get_vacuum_energies(calc_mace_off, calc_mace_mp, all_zs_in_dataset)
    
    E_total = np.array([atoms.info['energy_mace_off'] for atoms in full_atoms_list])
    X_table = np.zeros((len(full_atoms_list), n_species), dtype=np.int64)
    for i, atoms in enumerate(full_atoms_list):
        for z in atoms.get_atomic_numbers():
            if z in z_map: X_table[i, z_map[z]] += 1
    epera_regression_shifts = compute_E_statistics_vectorized(
        E=E_total, N=None, X=X_table, n_species=n_species, 
        delta_vacuum_energies=vacuum_energies, z_map=z_map
    )
    shifts_for_model = torch.tensor(epera_regression_shifts, dtype=torch.float64, device=device)

    # --- 3. Create ORIGINAL Datasets and PyG Loaders ---
    train_dataset_pyg = AtomsDataset(training_atoms_list, r_max=r_max, z_map=z_map)
    val_dataset_pyg = AtomsDataset(validation_atoms_list, r_max=r_max, z_map=z_map)
    
    # Use a large batch size for pre-computation to speed it up
    pyg_train_loader = PyGDataLoader(train_dataset_pyg, batch_size=32, shuffle=False, collate_fn=pyg_collate)
    pyg_val_loader = PyGDataLoader(val_dataset_pyg, batch_size=32, shuffle=False, collate_fn=pyg_collate)

    # --- 4. Create Feature Extractor Model ---
    # We only use this model once for pre-computation
    feature_extractor = DualReadoutMACE(
        base_mace_model=base_mace_model, 
        # We don't need shifts here, as we will apply them in the *new* model
        vacuum_energy_shifts=None, 
        mlp_hidden_features="auto", # Doesn't matter, we don't use the readout
        mlp_activation=mlp_activation,
        use_pca=False, # We just want the raw features
        pca_variance_threshold=pca_variance_threshold
    ).to(device)
    
    # --- 5. RUN PRE-COMPUTATION ---
    (
        train_features, train_node_attrs, train_targets, 
        train_base_energies, train_true_final_energies, train_ptr
    ) = precompute_features_in_memory(feature_extractor, pyg_train_loader, device)
    
    (
        val_features, val_node_attrs, val_targets,
        val_base_energies, val_true_final_energies, val_ptr
    ) = precompute_features_in_memory(feature_extractor, pyg_val_loader, device)

    # We are done with the heavy model and PyG loaders
    del feature_extractor
    del base_mace_model
    del pyg_train_loader
    del pyg_val_loader
    torch.cuda.empty_cache() # Clear VRAM

    # --- 6. Create FAST Datasets and DataLoaders ---
    fast_train_dataset = FastCachedDataset(
        train_features, train_node_attrs, train_targets, 
        train_base_energies, train_true_final_energies, train_ptr
    )
    fast_val_dataset = FastCachedDataset(
        val_features, val_node_attrs, val_targets, 
        val_base_energies, val_true_final_energies, val_ptr
    )
    
    # Use standard, much faster DataLoader
    train_loader = StandardDataLoader(
        fast_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=fast_collate_fn
    )
    val_loader = StandardDataLoader(
        fast_val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=fast_collate_fn
    )
    logging.info(f"Fast training set: {len(fast_train_dataset)}, Fast validation set: {len(fast_val_dataset)}")

    # --- 7. Create and Finalize the LIGHTWEIGHT Model ---
    model = CachedReadoutModel(
        in_features=train_features.shape[1],
        vacuum_energy_shifts=shifts_for_model, # Pass the shifts here
        mlp_hidden_features=mlp_hidden_features,
        mlp_activation=mlp_activation,
        use_pca=use_pca,
        pca_variance_threshold=pca_variance_threshold,
        z_map=z_map
    ).to(device)

    # Finalize the model using the training features
    model.finalize_model(train_features.to(device))
    print_model_summary(model) # Print summary of the *new* lightweight model

    # --- 8. Create Optimizer and Loss Function ---
    # Optimizer now targets the *lightweight* model's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    loss_fn = DeltaEnergyLoss() # This loss function works perfectly with our new model's output dict

    # --- 9. Start FAST Training Loop ---
    logging.info("Starting fast training on cached features...")
    best_val_mae = float('inf')
    
    # --- Early Stopping Parameters ---
    epochs_no_improve = 100
    early_stop_patience = 10
    early_stop_threshold = 1e-5
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # This loop is now extremely fast
        for batch in train_loader:
            # Move all tensors in the batch dictionary to the device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            
            output = model(batch)
            loss = loss_fn(output, batch)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch['num_graphs']

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # --- Use the new cached evaluation function ---
        avg_val_loss, delta_mae, delta_rmse, final_energy_mae_per_atom = evaluate_cached(
            model, val_loader, loss_fn, device
        )
        scheduler.step(final_energy_mae_per_atom)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val MAE (delta): {delta_mae:.6f} | "
            f"Val MAE (Final E/atom): {final_energy_mae_per_atom:.6f}"
        )
        
        # --- Check for early stopping and model saving ---
        improvement = best_val_mae - final_energy_mae_per_atom

        if improvement > early_stop_threshold:
            best_val_mae = final_energy_mae_per_atom
            epochs_no_improve = 0
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            # Save the lightweight model's state dict
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path} (Val MAE E/atom: {best_val_mae:.6f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            logging.info(f"Early stopping triggered: No significant improvement (> {early_stop_threshold}) for {early_stop_patience} epochs.")
            break # Exit the training loop

    logging.info("Training complete.")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))

    # --- Calculate and log total execution time ---
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Total execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
