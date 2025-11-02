# run_training_fast.py
import logging
import os
import random
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from ase import Atoms
from mace.calculators import MACECalculator

# Import all necessary components from your other files
from models import DualReadoutMACE, compute_E_statistics_vectorized, get_vacuum_energies, print_model_summary
from train import (
    evaluate,
    DeltaEnergyLoss,
    load_data,
    pyg_collate,
    AtomsDataset,
    evaluate_fast,
    precompute_dataset
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    training_dataset_path = "{training_dataset.xyz}"
    validation_dataset_path = "{validation_dataset.xyz}"
    base_model_path = "{base_model.model}"
    high_accuracy_model_path = "{hoght_model.model}"
    epochs = "{N_epochs}"
    lr = "{lr}"
    batch_size = 10
    output_dir = "delta_model_checkpoints"
    
    # --- MLP and PCA Configuration ---
    mlp_hidden_features = "{n_hidden}" # "auto" or an np.array 
    mlp_activation = {"activation_type"} #silu or swiglu
    use_pca = {"True"} #bool True or False
    pca_variance_threshold = {"threshold"}

    os.makedirs(output_dir, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logging.info(f"Using device: {device_str}")

    full_atoms_list = load_data(dataset_path)
    base_mace_model = torch.load(base_model_path, map_location=device)
    base_mace_model.to(dtype=torch.float64).eval()

    r_max = base_mace_model.r_max.item()
    atomic_numbers_list = base_mace_model.atomic_numbers.tolist()
    z_map = {z: i for i, z in enumerate(atomic_numbers_list)}
    n_species = len(atomic_numbers_list)

    # ... (Regression logic to get shifts remains the same) ...
    calc_mace_off = MACECalculator(
        model_path=high_accuracy_model_path, device=device_str
    )
    calc_mace_mp = MACECalculator(model_path=base_model_path, device=device_str)
    all_zs_in_dataset = [
        z for atoms in full_atoms_list for z in atoms.get_atomic_numbers()
    ]
    vacuum_energies = get_vacuum_energies(
        calc_mace_off, calc_mace_mp, all_zs_in_dataset
    )
    E_total = np.array([atoms.info["energy_mace_off"] for atoms in full_atoms_list])
    X_table = np.zeros((len(full_atoms_list), n_species), dtype=np.int64)
    for i, atoms in enumerate(full_atoms_list):
        for z in atoms.get_atomic_numbers():
            if z in z_map:
                X_table[i, z_map[z]] += 1
    epera_regression_shifts = compute_E_statistics_vectorized(
        E=E_total,
        N=None,
        X=X_table,
        n_species=n_species,
        delta_vacuum_energies=vacuum_energies,
        z_map=z_map,
    )
    shifts_for_model = torch.tensor(
        epera_regression_shifts, dtype=torch.float64, device=device
    )

    # --- 2. Create ORIGINAL Datasets and Dataloaders (for one-time use) ---
    train_dataset = AtomsDataset(training_atoms_list, r_max=r_max, z_map=z_map)
    val_dataset = AtomsDataset(validation_atoms_list, r_max=r_max, z_map=z_map)
    train_loader = PyGDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pyg_collate
    )
    val_loader = PyGDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate
    )
    logging.info(
        f"Original dataset: Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}"
    )

    # --- 3. Create the UNFINALIZED Model Shell ---
    model = DualReadoutMACE(
        base_mace_model=base_mace_model,
        vacuum_energy_shifts=shifts_for_model,
        mlp_hidden_features=mlp_hidden_features,
        mlp_activation=mlp_activation,
        use_pca=use_pca,
        pca_variance_threshold=pca_variance_threshold,
    ).to(device)

    # --- 4. Pre-compute Features for Train and Validation Sets ---
    precomputed_train_list = precompute_dataset(model, train_loader, device)
    precomputed_val_list = precompute_dataset(model, val_loader, device)

    # --- 5. Finalize the Model ---
    logging.info("Collecting training features for PCA fitting...")
    all_train_features = torch.cat(
        [data.x for data in precomputed_train_list], dim=0
    )
    logging.info(
        f"Collected {all_train_features.shape[0]} feature vectors for PCA."
    )

    model.finalize_model(all_train_features.to(device))
    print_model_summary(model)

    # --- 6. Create New Dataloaders from Pre-computed Data ---
    # We can now use a standard PyGDataLoader, no special collate_fn needed
    train_loader_fast = PyGDataLoader(
        precomputed_train_list, batch_size=batch_size, shuffle=True
    )
    val_loader_fast = PyGDataLoader(
        precomputed_val_list, batch_size=batch_size, shuffle=False
    )

    # Clean up memory
    del (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        precomputed_train_list,
        precomputed_val_list,
        all_train_features,
        full_atoms_list,
        dataset,
    )
    gc.collect()
    logging.info("Created fast dataloaders and cleaned up original data.")

    # --- 7. Create Optimizer AFTER Finalization ---
    optimizer = torch.optim.Adam(model.delta_readout.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )
    loss_fn = DeltaEnergyLoss()

    # --- 8. Start FAST Training Loop ---
    logging.info("Starting fast training loop...")
    best_val_mae = float("inf")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader_fast:  # Use the new loader
            batch = batch.to(device)
            optimizer.zero_grad()

            # --- Manually replicate the model's forward pass ---
            processed_features = batch.x
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

            batch_map = batch.batch
            delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)
            # --- End of manual forward pass ---

            output = {"delta_energy": delta_energy}  # All loss_fn needs
            loss = loss_fn(output, batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader_fast.dataset)

        # --- Evaluation ---
        # Use the new evaluate_fast function
        (
            avg_val_loss,
            delta_mae,
            delta_rmse,
            final_energy_mae_per_atom,
        ) = evaluate_fast(model, val_loader_fast, loss_fn, device)

        scheduler.step(final_energy_mae_per_atom)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val MAE (delta): {delta_mae:.6f} | "
            f"Val MAE (Final E/atom): {final_energy_mae_per_atom:.6f}"
        )

        if final_energy_mae_per_atom < best_val_mae:
            best_val_mae = final_energy_mae_per_atom
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                f"New best model saved to {checkpoint_path} (Val MAE E/atom: {best_val_mae:.6f})"
            )

    logging.info("Training complete.")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))


if __name__ == "__main__":
    main()
