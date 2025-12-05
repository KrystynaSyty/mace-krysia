import logging
import os
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as StandardDataLoader, Subset

from torch_geometric.loader import DataLoader as PyGDataLoader
from ase.io import read
from mace.calculators import MACECalculator
from ase.units import kJ, mol

# --- Import NEW Kernel Models ---
try:
    from models.kernel import CachedReadoutKernelModel, SpecificAtomKernelLayer
except ImportError:
    logging.error("Could not import kernel models. Make sure 'models_kernel.py' is in your path.")
    raise

# --- Import Original Models ---
try:
    from models.models import DualReadoutMACE, ScaleShiftBlock, PCALayer
    from models.models import compute_E_statistics_vectorized 
except ImportError:
     logging.error("Could not import DualReadoutMACE, ScaleShiftBlock or compute_E_statistics_vectorized.")
     raise

# --- Import Helpers ---
try:
    from models.helping_func import get_vacuum_energies, print_model_summary, generate_molecular_shift, check_feature_normalization
    from train.train import DeltaEnergyLoss, load_data, pyg_collate, AtomsDataset 
    from train.train_cached import fast_collate_fn, FastCachedDataset, precompute_features_in_memory, evaluate_cached
except ImportError:
    logging.error("Could not import helper functions from train/ or helping_func.py.")
    raise

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    start_time = time.time()
    EV_TO_KJ_MOL = 1.0 / (kJ/mol)  
    
    # --- Configuration ---
    logging.info("--- KERNEL MODEL: COARSE-TO-FINE SIGMA TUNING ---")
    
    cv_dataset_path = "subsample_md_finite_training.xyz" 
    test_dataset_path = "subsample_md_pbc_validation.xyz"
    
    base_model_path = "/mnt/storage_3/home/krystyna_syty/models/michaelides_2025/21_urea/MACE_model_swa.model"
    high_accuracy_model_path = "/mnt/storage_3/home/krystyna_syty/models/mace/MACE-omol-0-extra-large-1024.model"
    
    # --- HYPERPARAMETERS ---
    # 1. Lambda is fixed based on observation that it is less sensitive
    FIXED_LAMBDA = 1e-8 
    
    # 2. Coarse Search Config (Broad Sweep)
    COARSE_START_EXP = 0
    COARSE_END_EXP = -2 
    COARSE_STEPS = 5        # Number of steps in coarse grid
    
    # 3. Fine Search Config (Zoom In)
    FINE_STEPS = 10         # Resolution for the second pass
    
    # --- Config for splitting data ---
    fixed_val_split_size = 0.1 
    
    # --- BATCH SIZE ---
    PRECOMPUTE_BATCH_SIZE = 8 
    EVAL_BATCH_SIZE = 64      
    
    output_dir = "kernel_tuning_results"
    
    use_pca = True
    pca_variance_threshold = 1.0 
    random_seed = 42
    shift_type =  'molecular' 
    molecule_path = '/mnt/storage_3/home/krystyna_syty/pl0415-02/project_data/Krysia/mbe-automation/Systems/X23/21_urea/molecule.xyz'
    # --- End Configuration ---
    
    n_atoms_per_molecule = None 
    reference_molecule = None
    total_molecular_delta = 0.0 

    # --- Setup: Device, Directories, Shifts ---
    if shift_type != 'molecular':
        raise ValueError("This script is designed for 'molecular' shift type.")
        
    reference_molecule = read(molecule_path)
    n_atoms_per_molecule = len(reference_molecule) 
    logging.info(f"Using 'molecular' shift. Reference molecule: {reference_molecule.get_chemical_formula()} ({n_atoms_per_molecule} atoms)")

    os.makedirs(output_dir, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu" 
    device = torch.device(device_str)
    logging.info(f"Using device: {device_str}")

    # --- 1. Load Data ---
    logging.info(f"Loading CV data from: '{cv_dataset_path}'")
    cv_atoms_list = load_data(cv_dataset_path)
    logging.info(f"Loading TEST data from: '{test_dataset_path}'")
    test_atoms_list = load_data(test_dataset_path)

    # --- 2. Calculate Shifts AND Pre-process Data Targets ---
    base_mace_model = torch.load(base_model_path, map_location=device)
    base_mace_model.to(dtype=torch.float64).eval()
    r_max = base_mace_model.r_max.item()
    atomic_numbers_list = base_mace_model.atomic_numbers.tolist()
    z_map = {z: i for i, z in enumerate(atomic_numbers_list)}
    
    logging.info("Loading calculators for shift computation...")
    calc_mace_off = MACECalculator(model_paths=high_accuracy_model_path, device=device_str)
    calc_mace_mp = MACECalculator(model_paths=base_model_path, device=device_str)

    logging.info(f"Calculating 'molecular' shift...")
    total_molecular_delta = generate_molecular_shift(
        molecule=reference_molecule,
        high_accuracy_calculator=calc_mace_off,
        low_accuracy_calculator=calc_mace_mp
    )
    logging.info(f"Calculated TOTAL 'molecular' delta: {total_molecular_delta:.6f} eV")

    logging.info("Pre-processing data targets to be the 'residual' energy...")
    all_atoms_list = cv_atoms_list + test_atoms_list
    for atoms in all_atoms_list:
        n_mols = len(atoms) / n_atoms_per_molecule
        e_mol_shift = total_molecular_delta * n_mols
        e_total_delta = atoms.info['total_delta_energy']
        e_residual = e_total_delta - e_mol_shift
        atoms.info['y_target'] = e_residual 
    
    # --- Calculate Scaling ---
    atomic_inter_scale = 1.0
    atomic_inter_shift = 0.0 
 
    # --- 3. Create ORIGINAL Datasets and PyG Loaders ---
    logging.info(f"Creating PyG datasets...")
    cv_dataset_pyg = AtomsDataset(cv_atoms_list, r_max=r_max, z_map=z_map)
    test_dataset_pyg = AtomsDataset(test_atoms_list, r_max=r_max, z_map=z_map)

    pyg_cv_loader = PyGDataLoader(cv_dataset_pyg, batch_size=PRECOMPUTE_BATCH_SIZE, shuffle=False, collate_fn=pyg_collate)
    pyg_test_loader = PyGDataLoader(test_dataset_pyg, batch_size=PRECOMPUTE_BATCH_SIZE, shuffle=False, collate_fn=pyg_collate)
    
    # --- 4. Create Feature Extractor Model ---
    feature_extractor = DualReadoutMACE(
        base_mace_model=base_mace_model, 
        atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        mlp_hidden_features="auto", mlp_activation="silu", 
        use_pca=False, 
    ).to(device)
    
    # --- 5. RUN PRE-COMPUTATION (ONCE) ---
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

    # Combine for PCA fitting
    pca_features = torch.cat([cv_features, test_features], dim=0)
    
    del feature_extractor, base_mace_model, pyg_cv_loader, pyg_test_loader, calc_mace_off, calc_mace_mp
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
        fast_test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=fast_collate_fn
    )

    # --- Baseline Evaluation ---
    num_atoms_test = (test_ptr[1:] - test_ptr[:-1]).to(test_targets.device)
    baseline_mae_per_atom = torch.mean(torch.abs(test_targets) / num_atoms_test).item()
    
    logging.info("=" * 60)
    logging.info(f"BASELINE (SHIFT-ONLY) TEST MAE: {baseline_mae_per_atom:.6f} eV/atom")
    logging.info("=" * 60)

    # --- 7. Split CV data into Train and Fixed Validation ---
    all_indices = np.arange(len(fast_cv_dataset))
    train_indices, fixed_val_indices = train_test_split(
        all_indices, test_size=fixed_val_split_size, random_state=random_seed, shuffle=True
    )
    
    train_subset = Subset(fast_cv_dataset, train_indices)
    
    # Prepare Fitting Tensors
    temp_train_loader = StandardDataLoader(train_subset, batch_size=len(train_subset), shuffle=False, collate_fn=fast_collate_fn)
    train_data_batch = next(iter(temp_train_loader))
    del temp_train_loader 
    
    fit_features = train_data_batch['x'].to(device)
    fit_targets = train_data_batch['y'].to(device)
    fit_num_atoms = train_data_batch['num_atoms']
    fit_ptr = torch.cat([torch.tensor([0]), torch.cumsum(fit_num_atoms, dim=0)]).to(device)

    # --- Shared PCA Fitting ---
    logging.info("Fitting Shared Normalized PCA Layer...")
    if use_pca:
        shared_pca_layer = PCALayer(
            in_features=cv_features.shape[1],
            explained_variance_threshold=pca_variance_threshold
        ).to(device)
        shared_pca_layer.fit(pca_features.to(device))
    else:
        shared_pca_layer = None

    del pca_features
    torch.cuda.empty_cache()

    loss_fn = DeltaEnergyLoss(beta=0.1)

    # =========================================================================
    # --- HELPER FUNCTION FOR TRAINING ---
    # =========================================================================
    def run_sigma_eval(sigma_val, lambda_val):
        """Builds model, fits kernel, evaluates on test set."""
        model = CachedReadoutKernelModel(
            kernel_layer_class=SpecificAtomKernelLayer,
            in_features=cv_features.shape[1],
            atomic_inter_scale=atomic_inter_scale,
            atomic_inter_shift=atomic_inter_shift,
            use_pca=use_pca,
            pca_variance_threshold=pca_variance_threshold,
            kernel_sigma=sigma_val,
            kernel_lambda=lambda_val,
        ).to(device)
        
        if use_pca and shared_pca_layer is not None:
            model.pca_layer = shared_pca_layer
            model.pca_layer.fitted = True

        model.fit_kernel_head(fit_features, fit_targets, fit_ptr)
        
        (_, _, _, mae) = evaluate_cached(
            model, test_loader, loss_fn, device,
            total_molecular_delta=total_molecular_delta,
            n_atoms_per_molecule=n_atoms_per_molecule
        )
        return mae, model

    # =========================================================================
    # --- STAGE 1: COARSE GRID SEARCH ---
    # =========================================================================
    logging.info("=" * 60)
    logging.info(f"STAGE 1: COARSE SEARCH (10^{COARSE_START_EXP} to 10^{COARSE_END_EXP})")
    logging.info("=" * 60)
    
    coarse_sigmas = torch.logspace(COARSE_START_EXP, COARSE_END_EXP, steps=COARSE_STEPS).tolist()
    all_results = [] # List of dicts {'sigma': ..., 'mae': ...}

    for i, current_sigma in enumerate(coarse_sigmas):
        logging.info(f"[Coarse {i+1}/{len(coarse_sigmas)}] Testing Sigma={current_sigma:.2e}")
        mae, _ = run_sigma_eval(current_sigma, FIXED_LAMBDA)
        
        diff = baseline_mae_per_atom - mae
        logging.info(f"    -> MAE: {mae:.6f} (vs Baseline: {diff:+.6f})")
        all_results.append({'sigma': current_sigma, 'mae': mae, 'type': 'coarse'})

    # --- Analyze Coarse Results ---
    # Sort by MAE to find best
    sorted_by_mae = sorted(all_results, key=lambda x: x['mae'])
    best_coarse_sigma = sorted_by_mae[0]['sigma']
    best_coarse_mae = sorted_by_mae[0]['mae']
    
    logging.info(f"Best Coarse Sigma: {best_coarse_sigma:.2e} (MAE: {best_coarse_mae:.6f})")

    # Determine Bounds for Fine Search
    # 1. Sort results by Sigma value (descending: large sigma -> small sigma)
    coarse_by_sigma = sorted(all_results, key=lambda x: x['sigma'], reverse=True)
    
    # 2. Find index of the best sigma in the sorted list
    best_idx = -1
    for i, res in enumerate(coarse_by_sigma):
        if res['sigma'] == best_coarse_sigma:
            best_idx = i
            break
            
    # 3. Upper Bound: One step larger (index - 1)
    upper_idx = max(0, best_idx - 1)
    sigma_upper = coarse_by_sigma[upper_idx]['sigma']
    
    # 4. Lower Bound: Scan downwards (index + 1 onwards) until worse than baseline
    sigma_lower = coarse_by_sigma[-1]['sigma'] # Default to smallest
    found_cutoff = False
    
    for i in range(best_idx + 1, len(coarse_by_sigma)):
        current_sig_mae = coarse_by_sigma[i]['mae']
        # Check if "0 improvement" (MAE >= baseline)
        if current_sig_mae >= baseline_mae_per_atom:
            sigma_lower = coarse_by_sigma[i]['sigma']
            found_cutoff = True
            break
    
    if not found_cutoff:
        logging.info("Warning: Even smallest sigma improved over baseline. Lower bound set to minimum coarse sigma.")

    # =========================================================================
    # --- STAGE 2: FINE GRID SEARCH (ZOOM IN) ---
    # =========================================================================
    logging.info("=" * 60)
    logging.info(f"STAGE 2: FINE SEARCH")
    logging.info(f"Zooming in between {sigma_upper:.2e} and {sigma_lower:.2e}")
    logging.info("=" * 60)

    # Create log-spaced fine grid
    fine_sigmas = torch.logspace(
        np.log10(sigma_upper), 
        np.log10(sigma_lower), 
        steps=FINE_STEPS
    ).tolist()
    
    best_overall_mae = float('inf')
    best_overall_sigma = 0.0

    for i, current_sigma in enumerate(fine_sigmas):
        logging.info(f"[Fine {i+1}/{len(fine_sigmas)}] Testing Sigma={current_sigma:.4e}")
        mae, model = run_sigma_eval(current_sigma, FIXED_LAMBDA)
        
        diff = baseline_mae_per_atom - mae
        logging.info(f"    -> MAE: {mae:.6f} (vs Baseline: {diff:+.6f})")
        
        all_results.append({'sigma': current_sigma, 'mae': mae, 'type': 'fine'})

        if mae < best_overall_mae:
            best_overall_mae = mae
            best_overall_sigma = current_sigma
            logging.info(f"    *** NEW BEST FOUND! ***")
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(output_dir, "best_kernel_model.pt"))
        
        del model

    # --- Final Summary ---
    logging.info("=" * 60)
    logging.info("OPTIMIZATION COMPLETE")
    logging.info(f"Baseline MAE:        {baseline_mae_per_atom:.6f} eV/atom")
    logging.info(f"Best Sigma Found:    {best_overall_sigma:.4e}")
    logging.info(f"Best MAE Achieved:   {best_overall_mae:.6f} eV/atom")
    
    if best_overall_mae < baseline_mae_per_atom:
        logging.info(f"SUCCESS: Improved over baseline by {baseline_mae_per_atom - best_overall_mae:.6f} eV/atom")
    else:
        logging.warning("RESULT: No significant improvement over baseline found.")

    logging.info("-" * 60)
    logging.info("Full Results Summary:")
    
    # Sort ALL results by MAE
    all_results.sort(key=lambda x: x['mae'])
    
    # Header
    header = "{:<15} | {:<15} | {:<20} | {:<20}"
    logging.info(header.format("Sigma", "Lambda", "Test MAE (eV/atom)", "Improvement"))
    logging.info("-" * 80)
    
    for res in all_results:
        improvement = baseline_mae_per_atom - res['mae']
        logging.info(header.format(
            f"{res['sigma']:.2e}", 
            f"{FIXED_LAMBDA:.2e}", 
            f"{res['mae']:.6f}",
            f"{improvement:+.6f}"
        ))
    
    end_time = time.time()
    logging.info(f"Total Experiment Time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
