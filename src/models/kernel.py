import torch
import torch.nn as nn
import logging
from typing import Dict, Type
from torch_geometric.nn import global_add_pool
from tqdm import tqdm

# Import the helper modules from your existing 'models.py' file
try:
    from models.models import PCALayer, ScaleShiftBlock
except ImportError:
    try:
        from models.models import PCALayer, ScaleShiftBlock
    except ImportError:
        raise ImportError("Could not import PCALayer and ScaleShiftBlock. "
                          "Please ensure 'models.py' is accessible.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpecificAtomKernelLayer(nn.Module):
    """
    (To jest ta sama klasa co wcześniej, z dodanym paskiem postępu tqdm)
    Implementuje "specyficzną" atomistyczną warstwę KRR.
    """
    def __init__(self, sigma: float = 1.0, lambda_reg: float = 1e-10):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
            
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.fitted = False

        self.register_buffer('X_train', torch.empty(0, dtype=torch.get_default_dtype()))
        self.register_buffer('atomic_weights', torch.empty(0, dtype=torch.get_default_dtype())) 

    def _rbf_kernel_atomic(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(X1, X2, p=2).pow(2)
        return torch.exp(-dist_sq / (2.0 * self.sigma**2))

    def fit(self, 
            X_train_features: torch.Tensor, 
            y_train_energies: torch.Tensor, # This is now E_residual
            train_ptr: torch.Tensor):
        
        logging.info(f"Fitting Specific Atomistic Kernel on {len(y_train_energies)} structures ({X_train_features.shape[0]} atoms)...")
        device, dtype = X_train_features.device, X_train_features.dtype
        
        self.X_train = X_train_features.to(dtype)
        
        n_samples = len(y_train_energies) 
        n_atoms_total = self.X_train.shape[0]

        K_sum = torch.zeros((n_samples, n_samples), device=device, dtype=dtype)
        
        logging.info(f"  - Computing N_graphs x N_graphs 'sum' kernel (K_sum)...")
        # --- ADDED TQDM ---
        for i in tqdm(range(n_samples), desc="Building K_sum (i)"):
            atoms_i = self.X_train[train_ptr[i]:train_ptr[i+1]]
            for j in range(i, n_samples): 
                atoms_j = self.X_train[train_ptr[j]:train_ptr[j+1]]
                K_ij_atomic = self._rbf_kernel_atomic(atoms_i, atoms_j)
                k_val = torch.sum(K_ij_atomic)
                K_sum[i, j] = k_val
                K_sum[j, i] = k_val 

        logging.info("  - Solving linear system for structural weights 'c'...")
        y_total = y_train_energies.to(device=device, dtype=dtype) # Pass E_residual
        reg_matrix = torch.eye(n_samples, device=device, dtype=dtype) * self.lambda_reg
        
        try:
            c_weights = torch.linalg.solve(K_sum + reg_matrix, y_total) 
        except torch.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error: {e}. Matrix may be singular.")
            self.fitted = False
            return

        logging.info("  - Projecting to 'specific' atomic weights 'alpha'...")
        K_atom_struct = torch.zeros((n_atoms_total, n_samples), device=device, dtype=dtype)
        
        # --- ADDED TQDM ---
        for j in tqdm(range(n_atoms_total), desc="Projecting weights (j)"):
            atom_j = self.X_train[j:j+1] 
            for n in range(n_samples):
                atoms_n = self.X_train[train_ptr[n]:train_ptr[n+1]]
                k_j_n = self._rbf_kernel_atomic(atom_j, atoms_n)
                K_atom_struct[j, n] = torch.sum(k_j_n)
        
        self.atomic_weights = (K_atom_struct @ c_weights).unsqueeze(-1)
        self.fitted = True
        logging.info(f"Fit complete. Stored {self.atomic_weights.shape[0]} 'specific' atomic weights.")

    def forward(self, X_new: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("SpecificAtomKernelLayer has not been fitted. Call .fit(...) first.")
            
        K_new = self._rbf_kernel_atomic(X_new.to(self.X_train.dtype), self.X_train)
        per_atom_energy = K_new @ self.atomic_weights.to(K_new.dtype)
        
        return per_atom_energy # This is now the predicted atomic *residual*

    def __repr__(self):
        return (f"SpecificAtomKernelLayer(sigma={self.sigma}, lambda_reg={self.lambda_reg}, "
                f"fitted={self.fitted}, num_train_atoms={self.X_train.shape[0]})")


class CachedReadoutKernelModel(nn.Module):
    """
    --- V2: SIMPLIFIED MODEL ---
    A lightweight "cached" model that uses a Kernel Ridge Regression layer
    to predict ONLY the residual energy.
    """
    def __init__(
        self,
        in_features: int,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        use_pca: bool,
        pca_variance_threshold: float,
        
        # --- Kernel-specific Hyperparameters ---
        kernel_layer_class: Type[SpecificAtomKernelLayer],
        kernel_sigma: float,
        kernel_lambda: float,
        # --- 'atomic_energy_shifts' and 'z_map' are REMOVED (to fix your bug) ---
    ):
        super().__init__()
        self.use_pca = use_pca

        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=in_features,
                explained_variance_threshold=pca_variance_threshold
            )

        self.kernel_layer = kernel_layer_class(
            sigma=kernel_sigma, 
            lambda_reg=kernel_lambda
        )
            
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        # --- All shift-related buffers and logic are REMOVED ---
        
    def fit_kernel_head(self, 
                        all_features: torch.Tensor, 
                        all_targets: torch.Tensor, # This is E_residual
                        all_ptr: torch.Tensor):
        """
        Fits the KERNEL layer.
        ASSUMES `self.pca_layer.fit()` has been called separately.
        """
        features_to_fit = all_features
        device, dtype = all_features.device, all_features.dtype
        self.to(device=device, dtype=dtype)
        
        if self.use_pca:
            if not self.pca_layer.fitted:
                raise RuntimeError("PCA layer has not been fitted. Call `model.pca_layer.fit(global_pca_features)` first.")
            logging.info("Transforming training features with pre-fitted PCA...")
            features_to_fit = self.pca_layer(all_features)
        
        # Fit the kernel layer on the (possibly-PCA'd) features
        # all_targets is now E_residual
        self.kernel_layer.fit(features_to_fit, all_targets.to(dtype), all_ptr)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes a BATCH of pre-computed features and predicts the RESIDUAL delta.
        """
        if not self.kernel_layer.fitted:
            raise RuntimeError("Model is not fitted. Call .fit_kernel_head(...) first.")
        
        features = data['x']
        batch_map = data['batch_map']

        # 1. Apply PCA (if active)
        processed_features = self.pca_layer(features) if self.use_pca else features
        
        # 2. Get raw prediction from Kernel Layer
        atomic_residual_pred = self.kernel_layer(processed_features)
        
        # 3. Apply scale/shift
        scaled_atomic_residual = self.scale_shift(atomic_residual_pred)
        
        # 4. Sum *specific* per-atom residuals to get per-graph residual
        total_residual_pred = global_add_pool(scaled_atomic_residual, batch_map).squeeze(-1)
            
        # 5. Return *only* the predicted residual delta
        return {
            # This is the new output key, matching train.py and train_cached.py
            "delta_residual": total_residual_pred,
        }

    def __repr__(self):
        return (f"CachedReadoutKernelModel_V2_Simplified(\n"
                f"  use_pca={self.use_pca}\n"
                f"  pca_layer={self.pca_layer}\n"
                f"  kernel_layer={self.kernel_layer}\n"
                f"  scale_shift={self.scale_shift}\n"
                f")")
