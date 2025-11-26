import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as StandardDataLoader, Subset

from torch_geometric.loader import DataLoader as PyGDataLoader
from ase.io import read
from mace.calculators import MACECalculator
from ase.units import kJ,mol

# --- Import NEW Kernel Models ---
from models.kernel import CachedReadoutKernelModel, SpecificAtomKernelLayer
# --- Import Original Models (for feature extraction) ---
from models.models import DualReadoutMACE, compute_E_statistics_vectorized
# --- Import Helpers ---
from models.helping_func import get_vacuum_energies, print_model_summary, generate_molecular_shift
from train.train import DeltaEnergyLoss, load_data, pyg_collate, AtomsDataset 
from train.train_cached import fast_collate_fn, FastCachedDataset, precompute_features_in_memory, evaluate_cached

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    start_time = time.time()
    EV_TO_KJ_MOL = 1.0 / (kJ/mol)  
    
    # --- Configuration ---
    logging.info("--- KERNEL MODEL 'TRAINING' SCRIPT ---")
    
    cv_dataset_path = "subsumple_finite_training_dataset.xyz" 
    test_dataset_path = "subsumple_pbc_validation_dataset.xyz"
    
    high_accuracy_model_path = "/mnt/storage_3/home/krystyna_syty/models/michaelides_2025/08_cyanamide/MACE_model_swa.model"
    base_model_path = "/mnt/storage_3/home/krystyna_syty/models/mace/MACE-omol-0-extra-large-1024.model"
    
    # --- Kernel Hyperparameters (NEEDS TUNING!) ---
    # These replace 'total_epochs', 'lr', etc.
    KERNEL_SIGMA = 5.0     # The width of the Gaussian kernel
    KERNEL_LAMBDA = 1e-7   # The regularization strength
    
    # --- Config for splitting data ---
    fixed_val_split_size = 0.1 # We'll fit on 90% and validate on 10%
    
    batch_size = 64 # Only used for evaluation, so can be larger
    output_dir = "kernel_model_results"
    
    # --- Config for features (same as before) ---
    use_pca = True
    pca_variance_threshold = 1.0 # 1.0 means no reduction, just rotation/scaling
    random_seed = 42

    shift_type =  'molecular'
    molecule_path = '/mnt/storage_3/home/krystyna_syty/pl0415-02/project_data/Krysia/mbe-automation/Systems/X23/08_cyanamide/molecule.xyz'
    # --- End Configuration ---
    
    n_atoms_per_molecule = None 
    reference_molecule = None

    # (This section is identical to your training_ema.py)
    # --- Setup: Device, Directories, Shifts ---
    if shift_type == 'molecular':
        if not molecule_path or not isinstance(molecule_path, str):
            raise TypeError("Please define 'molecule_path' as a valid string path.")
        reference_molecule = read(molecule_path)
        n_atoms_per_molecule = len(reference_molecule) 
        logging.info(f"Using 'molecular' shift. Reference molecule: {reference_molecule.get_chemical_formula()} ({n_atoms_per_molecule} atoms)")
    else:
        logging.info(f"Using '{shift_type}' shift.")

    os.makedirs(output_dir, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu" 
    device = torch.device(device_str)
    logging.info(f"Using device: {device_str}")

    # --- 1. Load Data ---
    logging.info(f"Loading CV data from: '{cv_dataset_path}'")
    cv_atoms_list = load_data(cv_dataset_path)
    logging.info(f"Loading TEST data from: '{test_dataset_path}'")
    test_atoms_list = load_data(test_dataset_path)
    logging.info(f"Total structures for CV/Train: {len(cv_atoms_list)}")
    logging.info(f"Total structures for Test: {len(test_atoms_list)}")
    
    all_atoms_list_for_pca = cv_atoms_list + test_atoms_list
    logging.info(f"Total structures for PCA fitting: {len(all_atoms_list_for_pca)}")

    # --- 2. Load Base Model and Calculate Shifts (Identical) ---
    base_mace_model = torch.load(base_model_path, map_location=device)
    base_mace_model.to(dtype=torch.float64).eval()
    
    r_max = base_mace_model.r_max.item()
    atomic_numbers_list = base_mace_model.atomic_numbers.tolist()
    z_map = {z: i for i, z in enumerate(atomic_numbers_list)}
    n_species = len(atomic_numbers_list)

    calc_mace_off = MACECalculator(model_paths=high_accuracy_model_path, device=device_str)
    calc_mace_mp = MACECalculator(model_paths=base_model_path, device=device_str)
    all_zs_in_dataset = [z for atoms in (cv_atoms_list + test_atoms_list) for z in atoms.get_atomic_numbers()]
    
    atomic_shifts_for_model = None
    molecular_shift_for_model = None 

    logging.info("Calculating 'atomic' shifts using ONLY CV data...")
    vacuum_energies = get_vacuum_energies(calc_mace_off, calc_mace_mp, all_zs_in_dataset)
    
    E_total_cv = np.array([atoms.info['total_delta_energy'] for atoms in cv_atoms_list])
    cv_atom_counts = np.array([len(atoms) for atoms in cv_atoms_list], dtype=np.int64)
    
    X_table_cv = np.zeros((len(cv_atoms_list), n_species), dtype=np.int64)
    for i, atoms in enumerate(cv_atoms_list):
        for z in atoms.get_atomic_numbers():
            if z in z_map: X_table_cv[i, z_map[z]] += 1
            
    epera_regression_shifts = compute_E_statistics_vectorized(
        E=E_total_cv, N=None, X=X_table_cv, n_species=n_species, 
        delta_vacuum_energies=vacuum_energies, z_map=z_map
    )
    atomic_shifts_for_model = torch.tensor(epera_regression_shifts, dtype=torch.float64, device=device)
    logging.info("'Atomic' (regression) shifts calculated.")

    if shift_type == 'molecular':
        logging.info(f"Calculating 'molecular' shift using reference molecule ({reference_molecule.get_chemical_formula()})...")
        total_molecular_delta = generate_molecular_shift(
            molecule=reference_molecule,
            high_accuracy_calculator=calc_mace_off,
            low_accuracy_calculator=calc_mace_mp
        )
        molecular_shift_for_model = total_molecular_delta / n_atoms_per_molecule
        logging.info(f"Calculated PER-ATOM 'molecular' shift: {molecular_shift_for_model:.6f} eV")

    # --- Calculate Scaling (Identical) ---
    logging.info(f"Calculating standard deviation for scaling based on selected shift_type='{shift_type}'...")
    if shift_type == 'atomic':
        E_baseline_shift_cv = X_table_cv @ epera_regression_shifts
    elif shift_type == 'molecular':
        E_baseline_shift_cv = molecular_shift_for_model * cv_atom_counts
    else: # 'none'
        E_baseline_shift_cv = np.zeros_like(E_total_cv)

    E_residual_total_cv = E_total_cv - E_baseline_shift_cv
    E_residual_per_atom_cv = E_residual_total_cv / cv_atom_counts
    atomic_delta_std = np.std(E_residual_per_atom_cv)
    
    if atomic_delta_std < 1e-9:
        logging.warning(f"Calculated std dev is very small ({atomic_delta_std:.2e}). Setting scale to 1.0.")
        atomic_inter_scale = 1.0
    else:
        atomic_inter_scale = 1.0 / atomic_delta_std 
    atomic_inter_shift = 0.0 
    logging.info(f"Using atomic_inter_scale: {atomic_inter_scale:.6f}, atomic_inter_shift: {atomic_inter_shift:.6f}")
 
    # --- 3. Create ORIGINAL Datasets and PyG Loaders (for feature extraction) ---
    cv_dataset_pyg = AtomsDataset(cv_atoms_list, r_max=r_max, z_map=z_map)
    test_dataset_pyg = AtomsDataset(test_atoms_list, r_max=r_max, z_map=z_map)
    pca_dataset_pyg = AtomsDataset(all_atoms_list_for_pca, r_max=r_max, z_map=z_map)

    pyg_cv_loader = PyGDataLoader(cv_dataset_pyg, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate)
    pyg_test_loader = PyGDataLoader(test_dataset_pyg, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate)
    pyg_pca_loader = PyGDataLoader(pca_dataset_pyg, batch_size=batch_size, shuffle=False, collate_fn=pyg_collate)

    # --- 4. Create Feature Extractor Model ---
    feature_extractor = DualReadoutMACE(
        base_mace_model=base_mace_model, 
        atomic_energy_shifts=None, 
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        mlp_hidden_features="auto", mlp_activation="silu", # Dummies, not used
        use_pca=False, # PCA is handled by the kernel model
    ).to(device)
    
    # --- 5. RUN PRE-COMPUTATION ---
    logging.info("Pre-computing features for CV dataset...")
    (
        cv_features, cv_node_attrs, cv_targets, 
        cv_base_energies, cv_true_final_energies, cv_ptr
    ) = precompute_features_in_memory(feature_extractor, pyg_cv_loader, device)
    
    logging.info("Pre-computing features for TEST dataset...")
    (
        test_features, test_node_attrs, test_targets,
        test_base_energies, test_true_final_energies, test_ptr
    ) = precompute_features_in_memory(feature_extractor, pyg_test_loader, device)

    logging.info("Pre-computing features for PCA dataset (CV + Test)...")
    (
        pca_features, _, _, _, _, _
    ) = precompute_features_in_memory(feature_extractor, pyg_pca_loader, device)

    del feature_extractor, base_mace_model, pyg_cv_loader, pyg_test_loader, pyg_pca_loader
    torch.cuda.empty_cache()

    # --- 6. Create FAST Datasets ---
    fast_cv_dataset = FastCachedDataset(
        cv_features, cv_node_attrs, cv_targets, 
        cv_base_energies, cv_true_final_energies, cv_ptr
    )
    fast_test_dataset = FastCachedDataset(
        test_features, test_node_attrs, test_targets, 
        test_base_energies, test_true_final_energies, test_ptr
    )
    test_loader = StandardDataLoader(
        fast_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=fast_collate_fn
    )

    # --- 7. Split CV data into Train and Fixed Validation ---
    all_indices = np.arange(len(fast_cv_dataset))
    train_indices, fixed_val_indices = train_test_split(
        all_indices,
        test_size=fixed_val_split_size,
        random_state=random_seed,
        shuffle=True
    )
    
    # We need to manually re-build the tensors for the "fit" step
    train_subset = Subset(fast_cv_dataset, train_indices)
    
    # Create a temporary loader to re-batch the training subset
    temp_train_loader = StandardDataLoader(train_subset, batch_size=len(train_subset), shuffle=False, collate_fn=fast_collate_fn)
    train_data_batch = next(iter(temp_train_loader))
    
    fit_features = train_data_batch['x']
    fit_targets = train_data_batch['y']
    fit_num_atoms = train_data_batch['num_atoms']
    fit_ptr = torch.cat([torch.tensor([0]), torch.cumsum(fit_num_atoms, dim=0)])

    logging.info(f"Split CV data: {len(train_indices)} for fitting, {len(fixed_val_indices)} for fixed validation.")

    # Create the loader for the fixed validation set
    fixed_val_subset = Subset(fast_cv_dataset, fixed_val_indices)
    fixed_val_loader = StandardDataLoader(
        fixed_val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=fast_collate_fn
    )

    # --- 8. Create and "FIT" the KERNEL Model ---
    logging.info("Initializing the KERNEL model...")
    
    kernel_model_args = {
        'in_features': cv_features.shape[1],
        'atomic_energy_shifts': atomic_shifts_for_model,
        'molecular_energy_shift': molecular_shift_for_model,
        'n_atoms_per_molecule': n_atoms_per_molecule,
        'atomic_inter_scale': atomic_inter_scale,
        'atomic_inter_shift': atomic_inter_shift,
        'use_pca': use_pca,
        'pca_variance_threshold': pca_variance_threshold,
        'z_map': z_map,
        'kernel_sigma': KERNEL_SIGMA,
        'kernel_lambda': KERNEL_LAMBDA,
    }
    
    model = CachedReadoutKernelModel(
        kernel_layer_class=SpecificAtomKernelLayer,
        **kernel_model_args
    ).to(device)

    # --- This is the "TRAINING" step ---
    logging.info(f"Fitting kernel model with sigma={KERNEL_SIGMA}, lambda={KERNEL_LAMBDA}...")
    logging.info(f"Fitting PCA layer on ALL {len(pca_features)} CV + Test samples...")
    # First, fit PCA on *all* data to get a consistent transform
    model.pca_layer.fit(pca_features.to(device))
    
    logging.info(f"Fitting Kernel layer on {len(fit_features)} training samples...")
    # Now, fit the kernel on the *training* data only
    model.fit_kernel_head(
        fit_features.to(device), 
        fit_targets.to(device), 
        fit_ptr.to(device)
    )
    # --- "TRAINING" is complete ---
    
    print_model_summary(model)
    
    # --- 9. Define Loss and Run "Prediction Check" (from your script) ---
    loss_fn = DeltaEnergyLoss(beta=0.1) # Use SmoothL1Loss

    logging.info("Getting a fixed batch for prediction checking (from validation set)...")
    try:
        fixed_check_batch = next(iter(fixed_val_loader))
        fixed_check_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fixed_check_batch.items()}
        num_atoms_check_struct_0 = torch.sum(fixed_check_batch['batch_map'] == 0).item()

        model.eval()
        with torch.no_grad():
            output = model(fixed_check_batch)

        pred_total_energy = output['energy'][0].item()
        pred_energy_per_atom = pred_total_energy / num_atoms_check_struct_0
        
        true_total_energy = fixed_check_batch['true_final_energy'][0].item()
        true_energy_per_atom = true_total_energy / num_atoms_check_struct_0
        
        shift_only_pred_total_energy = float('nan') 
        if model.shift_type == 'molecular' and model.molecular_energy_shift is not None:
            base_energy_struct_0 = fixed_check_batch['base_energy'][0].item()
            total_shift = model.molecular_energy_shift * num_atoms_check_struct_0
            shift_only_pred_total_energy = base_energy_struct_0 + total_shift
        
        logging.info(f"--- KERNEL Model Prediction Check (Struct 0, {num_atoms_check_struct_0} atoms) ---")
        logging.info(f"    Total E (True):            {true_total_energy:16.6f} eV")
        logging.info(f"    Total E (Model Pred):        {pred_total_energy:16.6f} eV")
        if not np.isnan(shift_only_pred_total_energy):
            logging.info(f"    Total E (Shift-Only Pred):   {shift_only_pred_total_energy:16.6f} eV")
            logging.info(f"    Error (Model vs True):       {(pred_total_energy - true_total_energy):16.6f} eV")
            logging.info(f"    Error (Shift-Only vs True):  {(shift_only_pred_total_energy - true_total_energy):16.6f} eV")
        
        logging.info(
            f"    E/atom (Pred): {pred_energy_per_atom:12.6f} eV ({pred_energy_per_atom * EV_TO_KJ_MOL:12.4f} kJ/mol) | "
            f"E/atom (True): {true_energy_per_atom:12.6f} eV ({true_energy_per_atom * EV_TO_KJ_MOL:12.4f} kJ/mol)"
        )
    except StopIteration:
        logging.warning("Could not get a check batch, fixed validation set might be empty.")

    # --- 10. Final Evaluation on Fixed Validation Set ---
    logging.info("Evaluating fitted kernel model on FIXED validation set...")
    (val_loss, val_delta_mae, val_delta_rmse, val_final_mae) = evaluate_cached(
        model, fixed_val_loader, loss_fn, device
    )
    
    logging.info(
        f"--- KERNEL VALIDATION SET RESULTS (sigma={KERNEL_SIGMA}, lambda={KERNEL_LAMBDA}) ---"
    )
    logging.info(f"Val MAE (Final E/atom, eV/atom):   {val_final_mae:12.6f}")
    logging.info(f"Val MAE (Final E/atom, kJ/mol): {val_final_mae * EV_TO_KJ_MOL:12.6f}")

    # --- 11. Final Evaluation on TEST Set ---
    logging.info("Evaluating fitted kernel model on TEST set...")
    (test_loss, test_delta_mae, test_delta_rmse, test_final_mae_per_atom) = evaluate_cached(
        model, test_loader, loss_fn, device
    )
    
    logging.info(
        f"--- KERNEL TEST SET RESULTS (sigma={KERNEL_SIGMA}, lambda={KERNEL_LAMBDA}) ---"
    )
    logging.info(f"Test Loss (eV): {test_loss:12.6f}")
    logging.info(f"Test MAE (delta E, eV): {test_delta_mae:12.6f}")
    logging.info(f"Test MAE (Final E/atom, eV/atom):   {test_final_mae_per_atom:12.6f}")
    logging.info(f"Test MAE (Final E/atom, kJ/mol): {test_final_mae_per_atom * EV_TO_KJ_MOL:12.6f}")
    
    # --- Save the final model ---
    checkpoint_path = os.path.join(output_dir, f"kernel_model_s{KERNEL_SIGMA}_l{KERNEL_LAMBDA}.pt") 
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Saved final kernel model to {checkpoint_path}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Total execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

if __name__ == "__main__":
    main()
