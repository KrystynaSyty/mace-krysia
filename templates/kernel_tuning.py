import logging
import os
import time
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
    from models.models import DualReadoutMACE, ScaleShiftBlock 
    from models.models import compute_E_statistics_vectorized 
except ImportError:
     logging.error("Could not import DualReadoutMACE, ScaleShiftBlock or compute_E_statistics_vectorized.")
     raise

# --- Import Helpers ---
try:
    from models.helping_func import get_vacuum_energies, print_model_summary, generate_molecular_shift
    from train.train import DeltaEnergyLoss, load_data, pyg_collate, AtomsDataset 
    from train.train_cached import fast_collate_fn, FastCachedDataset, precompute_features_in_memory, evaluate_cached
except ImportError:
    logging.error("Could not import helper functions from train/ or helping_func.py.")
    raise

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CUSTOM PCA LAYER (With Normalization) ---
class PCALayer(nn.Module):
    """
    A non-trainable PCA layer that dynamically determines the number of components
    needed to retain a certain amount of variance.
    INCLUDES L2 NORMALIZATION in the forward pass.
    """
    def __init__(self, in_features: int, explained_variance_threshold: float = 0.99):
        super().__init__()
        if not (0 < explained_variance_threshold <= 1):
            raise ValueError("explained_variance_threshold must be between 0 and 1.")
            
        self.in_features = in_features
        self.explained_variance_threshold = explained_variance_threshold
        self.n_components = None  # Will be determined during fitting
        self.fitted = False

        # Buffers will be created dynamically in the .fit() method
        self.register_buffer('mean', torch.empty(0))
        self.register_buffer('components', torch.empty(0))
        self.register_buffer('scales', torch.empty(0))

    def fit(self, x: torch.Tensor):
        """
        Computes the number of components from the threshold and fits the PCA transformation.
        """
        logging.info(
            f"Fitting PCA layer to retain {self.explained_variance_threshold:.2%} of variance..."
        )
        if x.shape[1] != self.in_features:
            raise ValueError("Input data for fitting has incorrect number of features.")

        # --- Determine number of components ---
        x_centered_for_svd = x - torch.mean(x, dim=0)
        _, S, V = torch.linalg.svd(x_centered_for_svd, full_matrices=False)
        
        explained_variance = S.pow(2)
        cumulative_variance_ratio = torch.cumsum(explained_variance / torch.sum(explained_variance), dim=0)
        
        n_components = torch.searchsorted(
            cumulative_variance_ratio,
            torch.tensor(self.explained_variance_threshold, device=x.device, dtype=x.dtype)
        ).item() + 1
        self.n_components = n_components
        logging.info(f"Determined that {self.n_components} components are needed.")

        # --- Fit the transformation using the determined n_components ---
        self.mean = torch.mean(x, dim=0)
        self.components = V.T[:, :self.n_components]
        
        n_samples = x.shape[0]
        scales = S[:self.n_components] / np.sqrt(max(1, n_samples - 1))
        scales[scales == 0] = 1.0
        self.scales = scales

        self.fitted = True
        logging.info("PCA layer has been successfully fitted.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("PCA layer has not been fitted yet. Call .fit(data) before using.")
        
        x_centered = x - self.mean
        x_projected = x_centered @ self.components
        x_standardized = x_projected / self.scales
        
        # --- NORMALIZATION ADDED HERE ---
        x_normalized = F.normalize(x_standardized, p=2, dim=1)
        
        return x_normalized

def check_feature_normalization(features: torch.Tensor, tolerance: float = 1e-4):
    """
    Checks if the feature vectors are normalized to length 1 (Unit Norm).
    """
    logging.info("--- Checking Feature Normalization (Post-PCA) ---")
    norms = torch.linalg.norm(features, dim=1)
    mean_norm = torch.mean(norms).item()
    max_norm = torch.max(norms).item()
    min_norm = torch.min(norms).item()
    
    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
    
    logging.info(f"    Mean Norm: {mean_norm:.6f}")
    logging.info(f"    Max Norm:  {max_norm:.6f}")
    logging.info(f"    Min Norm:  {min_norm:.6f}")
    
    if is_normalized:
        logging.info("    [OK] Features are effectively normalized to 1.")
    else:
        logging.warning("    [WARNING] Features are NOT normalized to 1. This may affect Kernel performance.")
    logging.info("-------------------------------------------------")
    return is_normalized


def main():
    start_time = time.time()
    EV_TO_KJ_MOL = 1.0 / (kJ/mol)  
    
    # --- Configuration ---
    logging.info("--- KERNEL MODEL GRID SEARCH (Sigma & Lambda) ---")
    
    cv_dataset_path = "subsumple_finite_training_dataset.xyz" 
    test_dataset_path = "subsumple_pbc_validation_dataset.xyz"
    
    base_model_path = "/mnt/storage_3/home/krystyna_syty/models/michaelides_2025/08_cyanamide/MACE_model_swa.model"
    high_accuracy_model_path = "/mnt/storage_3/home/krystyna_syty/models/mace/MACE-omol-0-extra-large-1024.model"
    
    # --- HYPERPARAMETER GRID ---
    # Log-spaced sigmas. Since we normalize to 1, efficient sigmas are usually 0.1 - 2.0
    SIGMA_RANGE = torch.logspace(np.log10(3.91e-01), np.log10(2.55e-01), steps=10).tolist() # Testing 10.0 down to 0.01
    
    # Log-spaced lambda (regularization). 
    LAMBDA_RANGE = torch.logspace(-6, -8, steps=2).tolist() # Testing 1e-6 down to 1e-12
    
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
    molecule_path = '/mnt/storage_3/home/krystyna_syty/pl0415-02/project_data/Krysia/mbe-automation/Systems/X23/08_cyanamide/molecule.xyz'
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
    
    # Cleanup heavy models
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

    # --- NEW: Baseline (Shift-Only) Evaluation ---
    num_atoms_test = (test_ptr[1:] - test_ptr[:-1]).to(test_targets.device)
    baseline_mae_per_atom = torch.mean(torch.abs(test_targets) / num_atoms_test).item()
    
    logging.info("=" * 60)
    logging.info(f"BASELINE (SHIFT-ONLY) TEST MAE: {baseline_mae_per_atom:.6f} eV/atom")
    logging.info(f"BASELINE (SHIFT-ONLY) TEST MAE: {baseline_mae_per_atom * EV_TO_KJ_MOL:.6f} kJ/mol")
    logging.info("=" * 60)

    # --- 7. Split CV data into Train and Fixed Validation ---
    all_indices = np.arange(len(fast_cv_dataset))
    train_indices, fixed_val_indices = train_test_split(
        all_indices,
        test_size=fixed_val_split_size,
        random_state=random_seed,
        shuffle=True
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

    fixed_val_subset = Subset(fast_cv_dataset, fixed_val_indices)
    fixed_val_loader = StandardDataLoader(
        fixed_val_subset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=fast_collate_fn
    )

    # --- NEW: Shared PCA Fitting using local PCALayer ---
    logging.info("Fitting Shared Normalized PCA Layer...")
    
    # Instantiate our Custom Normalized Layer
    if use_pca:
        shared_pca_layer = PCALayer(
            in_features=cv_features.shape[1],
            explained_variance_threshold=pca_variance_threshold
        ).to(device)
        
        logging.info("Fitting PCA...")
        shared_pca_layer.fit(pca_features.to(device))
        
        # --- CHECK NORMALIZATION HERE ---
        logging.info("Verifying normalization on a sample subset...")
        with torch.no_grad():
            sample_features = fit_features[:1000] # Check first 1000 atoms
            # PCALayer.forward() will auto-normalize now
            transformed_sample = shared_pca_layer(sample_features)
            check_feature_normalization(transformed_sample)
    else:
        shared_pca_layer = None

    del pca_features
    torch.cuda.empty_cache()

    # --- START TUNING LOOP ---
    total_iterations = len(SIGMA_RANGE) * len(LAMBDA_RANGE)
    logging.info("=" * 60)
    logging.info(f"Starting Grid Search. Testing {total_iterations} combinations.")
    logging.info(f"Sigma Range: {min(SIGMA_RANGE):.2e} - {max(SIGMA_RANGE):.2e}")
    logging.info(f"Lambda Range: {min(LAMBDA_RANGE):.2e} - {max(LAMBDA_RANGE):.2e}")
    logging.info("=" * 60)
    
    best_mae = float('inf')
    best_params = {}
    results = []

    loss_fn = DeltaEnergyLoss(beta=0.1) 
    
    iter_count = 0

    for current_sigma in SIGMA_RANGE:
        for current_lambda in LAMBDA_RANGE:
            iter_count += 1
            iter_start = time.time()
            logging.info(f"--- [Iter {iter_count}/{total_iterations}] Sigma={current_sigma:.2e}, Lambda={current_lambda:.2e} ---")
    
            # 1. Initialize Model
            model = CachedReadoutKernelModel(
                kernel_layer_class=SpecificAtomKernelLayer,
                in_features=cv_features.shape[1],
                atomic_inter_scale=atomic_inter_scale,
                atomic_inter_shift=atomic_inter_shift,
                use_pca=use_pca,
                pca_variance_threshold=pca_variance_threshold,
                kernel_sigma=current_sigma,
                kernel_lambda=current_lambda,
            ).to(device)
            
            # 2. Inject Pre-fitted NORMALIZED PCA
            if use_pca and shared_pca_layer is not None:
                model.pca_layer = shared_pca_layer
                model.pca_layer.fitted = True
    
            # 3. Fit Kernel Head
            model.fit_kernel_head(
                fit_features, 
                fit_targets, 
                fit_ptr
            )
            
            # 4. Evaluate
            (test_loss, test_delta_mae, test_delta_rmse, test_final_mae_per_atom) = evaluate_cached(
                model, test_loader, loss_fn, device,
                total_molecular_delta=total_molecular_delta,
                n_atoms_per_molecule=n_atoms_per_molecule
            )
            
            iter_time = time.time() - iter_start
            
            # Log Result
            improvement = baseline_mae_per_atom - test_final_mae_per_atom
            logging.info(f"    -> Test MAE (eV/atom): {test_final_mae_per_atom:.6f}")
            logging.info(f"    -> vs Baseline: {improvement:+.6f} eV/atom")
            
            results.append({
                'sigma': current_sigma,
                'lambda': current_lambda,
                'mae': test_final_mae_per_atom,
                'time': iter_time
            })
            
            # Save Best
            if test_final_mae_per_atom < best_mae:
                best_mae = test_final_mae_per_atom
                best_params = {'sigma': current_sigma, 'lambda': current_lambda}
                logging.info(f"    *** NEW BEST FOUND! (Sigma={current_sigma:.2e}, Lambda={current_lambda:.2e}) ***")
                
                # Save the best model
                checkpoint_path = os.path.join(output_dir, f"best_kernel_model.pt") 
                torch.save(model.state_dict(), checkpoint_path)
            
            # Cleanup
            del model
            # torch.cuda.empty_cache() 

    # --- Final Summary ---
    logging.info("=" * 60)
    logging.info("TUNING COMPLETE")
    logging.info(f"Baseline (Shift-Only) MAE: {baseline_mae_per_atom:.6f} eV/atom")
    logging.info(f"Best Grid Search MAE:      {best_mae:.6f} eV/atom")
    logging.info(f"Best Parameters:           Sigma={best_params['sigma']:.2e}, Lambda={best_params['lambda']:.2e}")
    
    if best_mae < baseline_mae_per_atom:
        logging.info(f"SUCCESS: Improved over baseline by {baseline_mae_per_atom - best_mae:.6f} eV/atom")
    else:
        logging.warning("RESULT: No significant improvement over baseline found. The residual might be noise or hard to learn.")

    logging.info("-" * 60)
    logging.info("Full Results Summary:")
    
    # Sort by MAE
    results.sort(key=lambda x: x['mae'])
    
    # Header
    header = "{:<15} | {:<15} | {:<20} | {:<20}"
    logging.info(header.format("Sigma", "Lambda", "Test MAE (eV/atom)", "Improvement"))
    logging.info("-" * 80)
    
    for res in results:
        improvement = baseline_mae_per_atom - res['mae']
        logging.info(header.format(
            f"{res['sigma']:.2e}", 
            f"{res['lambda']:.2e}", 
            f"{res['mae']:.6f}",
            f"{improvement:+.6f}"
        ))
    
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total Experiment Time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
