import logging
import os
import random
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from ase import Atoms
from mace.calculators import MACECalculator

# Import all necessary components from your other files
from models.models import DualReadoutMACE, compute_E_statistics_vectorized, print_model_summary, get_vacuum_energies
from train.train import evaluate, DeltaEnergyLoss, load_data, pyg_collate, AtomsDataset

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

    # --- 1. Load Data from Separate Files ---
    # MODIFICATION: Load training and validation data from their respective files.
    logging.info(f"Loading training data from: '{training_dataset_path}'")
    training_atoms_list = load_data(training_dataset_path)
    logging.info(f"Loading validation data from: '{validation_dataset_path}'")
    validation_atoms_list = load_data(validation_dataset_path)
    
    # Combine lists *only* for calculating shared statistics (shifts, etc.)
    full_atoms_list = training_atoms_list + validation_atoms_list
    logging.info(f"Total structures loaded for statistics calculation: {len(full_atoms_list)}")

    # --- 2. Load Base Model and Calculate Shifts (as before) ---
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

    # --- 3. Create Datasets and Dataloaders ---
    # MODIFICATION: Create two separate AtomsDataset objects instead of splitting one.
    train_dataset = AtomsDataset(training_atoms_list, r_max=r_max, z_map=z_map)
    val_dataset = AtomsDataset(validation_atoms_list, r_max=r_max, z_map=z_map)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pyg_collate)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate)
    logging.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 4. Create and Finalize the Model (as before) ---
    model = DualReadoutMACE(
        base_mace_model=base_mace_model, 
        vacuum_energy_shifts=shifts_for_model,
        mlp_hidden_features=mlp_hidden_features,
        mlp_activation=mlp_activation,
        use_pca=use_pca,
        pca_variance_threshold=pca_variance_threshold
    ).to(device)
    
    logging.info("Collecting features from the training set to finalize the model...")
    all_features = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')
            _ = model.mace_model(data_dict, compute_force=False)
            all_features.append(model.features.cpu())
            model.features = None 

    training_features_tensor = torch.cat(all_features, dim=0)
    logging.info(f"Collected and concatenated {training_features_tensor.shape[0]} feature vectors.")
    
    model.finalize_model(training_features_tensor.to(device))
    print_model_summary(model)

    # --- 5. Create Optimizer and Loss Function (as before) ---
    optimizer = torch.optim.Adam(model.delta_readout.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    loss_fn = DeltaEnergyLoss()

    # --- 6. Start Training Loop (as before) ---
    logging.info("Starting training...")
    best_val_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            data_dict = batch.to_dict()
            if 'pos' in data_dict:
                data_dict['positions'] = data_dict.pop('pos')
            
            output = model(data_dict)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        avg_val_loss, delta_mae, delta_rmse, final_energy_mae_per_atom = evaluate(model, val_loader, loss_fn, device)
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
            logging.info(f"New best model saved to {checkpoint_path} (Val MAE E/atom: {best_val_mae:.6f})")

    logging.info("Training complete.")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))

if __name__ == "__main__":
    main()
