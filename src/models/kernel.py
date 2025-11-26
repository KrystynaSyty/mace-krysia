import torch
import torch.nn as nn
import logging
from typing import Dict, Type
from torch_geometric.nn import global_add_pool

from models.models import PCALayer, ScaleShiftBlock


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpecificAtomKernelLayer(nn.Module):
    """
    Implements a "specific" atomistic Kernel Ridge Regression (KRR) layer.
    
    This model assumes E_total = sum(E_atomic) and solves for the weights
    alpha_j for *every atom* in the training set. This is a "specific"
    atomistic model, not an "average" one.

    This layer is not trained with gradient descent. You must call .fit()
    on the entire training dataset.
    
    Args:
        sigma (float): Kernel width (hyperparameter) for the RBF kernel.
        lambda_reg (float): Regularization strength (hyperparameter).
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

        # Buffers to store after fitting
        # X_train will store ALL atoms from the training set
        self.register_buffer('X_train', torch.empty(0, dtype=torch.get_default_dtype()))
        # atomic_weights will store one alpha_j weight for EACH atom
        self.register_buffer('atomic_weights', torch.empty(0, dtype=torch.get_default_dtype())) # This is 'alpha'

    def _rbf_kernel_atomic(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Computes the RBF (Gaussian) kernel matrix between two sets of atoms.
        k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        
        Args:
            X1 (Tensor): Shape [N, D_features]
            X2 (Tensor): Shape [M, D_features]
            
        Returns:
            Tensor: Shape [N, M]
        """
        # torch.cdist computes pairwise distances efficiently
        dist_sq = torch.cdist(X1, X2, p=2).pow(2)
        return torch.exp(-dist_sq / (2.0 * self.sigma**2))

    def fit(self, 
            X_train_features: torch.Tensor, 
            y_train_energies: torch.Tensor, 
            train_ptr: torch.Tensor):
        """
        Fits the KRR model using the *entire* training dataset.
        This solves for the "specific" atomic weights.
        
        Args:
            X_train_features (Tensor): All atom features [M_total_atoms, N_features]
            y_train_energies (Tensor): Per-graph TOTAL delta energies [N_graphs]
            train_ptr (Tensor): The 'ptr' array for the training set [N_graphs + 1]
        """
        logging.info(f"Fitting Specific Atomistic Kernel on {len(y_train_energies)} structures...")
        device, dtype = X_train_features.device, X_train_features.dtype
        
        # 1. Store the training atom features (needed for prediction)
        self.X_train = X_train_features.to(dtype)
        
        n_samples = len(y_train_energies) # Number of graphs
        n_atoms_total = self.X_train.shape[0]

        # We solve this problem using a "dual kernel" approach, which is
        # more memory-efficient. We first solve a (N_graphs x N_graphs)
        # system and then project the weights onto the (M_total_atoms).

        # 2. Compute the N_graphs x N_graphs "sum" kernel (K_sum)
        # K_nm = sum_{i in n} sum_{j in m} k(x_i, x_j)
        K_sum = torch.zeros((n_samples, n_samples), device=device, dtype=dtype)
        
        logging.info(f"  - Computing N_graphs x N_graphs 'sum' kernel (K_sum)...")
        for i in range(n_samples):
            atoms_i = self.X_train[train_ptr[i]:train_ptr[i+1]]
            for j in range(i, n_samples): # Symmetric matrix
                atoms_j = self.X_train[train_ptr[j]:train_ptr[j+1]]
                
                # K_ij_atomic is [n_atoms_i, n_atoms_j]
                K_ij_atomic = self._rbf_kernel_atomic(atoms_i, atoms_j)
                
                k_val = torch.sum(K_ij_atomic)
                K_sum[i, j] = k_val
                K_sum[j, i] = k_val # Exploit symmetry

        # 3. Solve for 'c' weights (per-graph)
        # We solve (K_sum + lambda*I)c = y_total
        logging.info("  - Solving linear system for structural weights 'c'...")
        y_total = y_train_energies.to(device=device, dtype=dtype)
        reg_matrix = torch.eye(n_samples, device=device, dtype=dtype) * self.lambda_reg
        
        try:
            c_weights = torch.linalg.solve(K_sum + reg_matrix, y_total) # Shape [N_graphs]
        except torch.linalg.LinAlgError as e:
            logging.error(f"Linear algebra error: {e}. Matrix may be singular.")
            logging.error("Try increasing 'lambda_reg' (regularization) or checking your features.")
            self.fitted = False
            return

        # 4. Project structural 'c' weights back to "specific" atomic 'alpha' weights
        # alpha_j = sum_{n in N_graphs} c_n * (sum_{i in n} k(x_i, x_j))
        # This is an (M_atoms x N_graphs) @ (N_graphs) operation.
        
        logging.info("  - Projecting to 'specific' atomic weights 'alpha'...")
        
        # K_atom_struct is (M_total_atoms, N_graphs)
        # K_atom_struct[j, n] = sum_{i in n} k(x_j, x_i)
        K_atom_struct = torch.zeros((n_atoms_total, n_samples), device=device, dtype=dtype)
        
        for j in range(n_atoms_total):
            atom_j = self.X_train[j:j+1] # Shape [1, N_features]
            for n in range(n_samples):
                atoms_n = self.X_train[train_ptr[n]:train_ptr[n+1]]
                
                # k(x_j, atoms_n) -> [1, n_atoms_in_n]
                k_j_n = self._rbf_kernel_atomic(atom_j, atoms_n)
                K_atom_struct[j, n] = torch.sum(k_j_n)

        # alpha = K_atom_struct @ c
        # Store as [M_total_atoms, 1] for matrix multiplication in forward pass
        self.atomic_weights = (K_atom_struct @ c_weights).unsqueeze(-1)
        
        self.fitted = True
        logging.info(f"Fit complete. Stored {self.atomic_weights.shape[0]} 'specific' atomic weights.")

    def forward(self, X_new: torch.Tensor) -> torch.Tensor:
        """
        Predicts the *specific* per-atom energy for a new batch of atoms.
        
        E_atom = sum_{j in M_train_atoms} alpha_j * k(x_new, x_j)
        
        Args:
            X_new (Tensor): New atom features [N_batch_atoms, N_features]
            
        Returns:
            Tensor: Predicted *specific* per-atom energy [N_batch_atoms, 1]
        """
        if not self.fitted:
            raise RuntimeError("SpecificAtomKernelLayer has not been fitted. Call .fit(...) first.")
            
        # 1. Compute kernel between new atoms and *all* training atoms
        # K_new shape: [N_batch_atoms, M_total_train_atoms]
        K_new = self._rbf_kernel_atomic(X_new.to(self.X_train.dtype), self.X_train)
        
        # 2. Predict per-atom energy: y_pred = K_new @ alpha
        # [N_batch_atoms, M_train] @ [M_train, 1] -> [N_batch_atoms, 1]
        per_atom_energy = K_new @ self.atomic_weights.to(K_new.dtype)
        
        return per_atom_energy

    def __repr__(self):
        return (f"SpecificAtomKernelLayer(sigma={self.sigma}, lambda_reg={self.lambda_reg}, "
                f"fitted={self.fitted}, num_train_atoms={self.X_train.shape[0]})")


class CachedReadoutKernelModel(nn.Module):
    """
    A lightweight "cached" model that uses a Kernel Ridge Regression layer
    (like SpecificAtomKernelLayer) on pre-computed features.
    
    This model must be "fitted" on the entire training set, not trained with
    an optimizer.
    """
    def __init__(
        self,
        in_features: int,
        atomic_energy_shifts: torch.Tensor,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        use_pca: bool,
        pca_variance_threshold: float,
        z_map: Dict[int, int], # z_map is needed for shift_type='atomic'
        
        # --- Kernel-specific Hyperparameters ---
        kernel_layer_class: Type[SpecificAtomKernelLayer],
        kernel_sigma: float,
        kernel_lambda: float,
        
        # --- Standard arguments (same as CachedReadoutModel) ---
        shift_type: str = 'atomic',
        molecular_energy_shift: float = None,
        n_atoms_per_molecule: int = None
    ):
        super().__init__()
        self.use_pca = use_pca
        self.shift_type = shift_type.lower()
        self.molecular_energy_shift = molecular_energy_shift
        self.n_atoms_per_molecule = n_atoms_per_molecule
        self.z_map = z_map
        self.n_species = len(z_map)
        
        if self.shift_type == 'molecular' and self.molecular_energy_shift is None:
            raise ValueError(
                "For shift_type='molecular' you have to initiate 'molecular_energy_shift' (as the pre-calculated per-atom shift value)."
            )

        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=in_features,
                explained_variance_threshold=pca_variance_threshold
            )

        # --- Use the Kernel Layer instead of MLPReadout ---
        self.kernel_layer = kernel_layer_class(
            sigma=kernel_sigma, 
            lambda_reg=kernel_lambda
        )
        
        if atomic_energy_shifts is not None:
            self.register_buffer('atomic_energy_shifts', atomic_energy_shifts)
        else:
            self.atomic_energy_shifts = 0
            
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        
    def fit_kernel_head(self, 
                        all_features: torch.Tensor, 
                        all_targets: torch.Tensor, 
                        all_ptr: torch.Tensor):
        """
        Replaces 'finalize_model'. This fits the PCA layer AND the kernel layer.
        
        Args:
            all_features (Tensor): ALL training features [N_total_atoms, N_features]
            all_targets (Tensor): ALL training targets (per-graph delta) [N_graphs]
            all_ptr (Tensor): ALL training ptr [N_graphs + 1]
        """
        features_to_fit = all_features
        device, dtype = all_features.device, all_features.dtype
        
        self.to(device=device, dtype=dtype)
        
        if self.use_pca:
            logging.info("Fitting PCA layer...")
            self.pca_layer.fit(all_features)
            logging.info("Transforming all features with PCA...")
            # Transform all features at once for kernel fitting
            features_to_fit = self.pca_layer(all_features)
        
        # Fit the kernel layer on the (possibly-PCA'd) features
        self.kernel_layer.fit(features_to_fit, all_targets.to(dtype), all_ptr)
        logging.info("Kernel readout head has been successfully fitted.")

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes a BATCH of pre-computed features and makes a prediction.
        
        Args:
            data (dict): A batch dictionary from 'fast_collate_fn' containing
                         'x', 'batch_map', 'base_energy', 'node_attrs'.
                         
        Returns:
            dict: {"energy": Tensor, "delta_energy": Tensor}
        """
        if not self.kernel_layer.fitted:
            raise RuntimeError("Model is not fitted. Call .fit_kernel_head(...) first.")
        
        features = data['x']
        batch_map = data['batch_map']
        base_energy = data['base_energy']

        # 1. Apply PCA (if active)
        processed_features = self.pca_layer(features) if self.use_pca else features
        
        # 2. Get raw prediction from Kernel Layer
        # This now returns *specific* atomic energies
        atomic_delta_prediction = self.kernel_layer(processed_features)
        
        # 3. Apply scale/shift
        scaled_atomic_delta = self.scale_shift(atomic_delta_prediction)
        
        # 4. Apply per-atom energy shifts (atomic or molecular)
        if self.shift_type == 'atomic':
            if isinstance(self.atomic_energy_shifts, torch.Tensor):
                # 'node_attrs' is [N_batch_atoms, N_species]
                try:
                    node_indices = torch.argmax(data['node_attrs'], dim=1)
                    shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
                    total_atomic_delta = scaled_atomic_delta + shifts.to(scaled_atomic_delta.dtype)
                except KeyError as e:
                    logging.error(f"Error getting node_attrs from batch: {e}. Batch keys: {data.keys()}")
                    total_atomic_delta = scaled_atomic_delta
            else:
                total_atomic_delta = scaled_atomic_delta
        
        elif self.shift_type == 'molecular':
            total_atomic_delta = scaled_atomic_delta + self.molecular_energy_shift
            
        else: # 'none'
            total_atomic_delta = scaled_atomic_delta

        # 5. Sum *specific* per-atom deltas to get per-graph delta
        delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)
        
        # 6. Reconstruct final energy
        final_energy = base_energy + delta_energy
            
        return {
            "energy": final_energy,
            "delta_energy": delta_energy
        }

    def __repr__(self):
        return (f"CachedReadoutKernelModel(\n"
                f"  use_pca={self.use_pca}, shift_type='{self.shift_type}'\n"
                f"  pca_layer={self.pca_layer}\n"
                f"  kernel_layer={self.kernel_layer}\n"
                f"  scale_shift={self.scale_shift}\n"
                f")")
