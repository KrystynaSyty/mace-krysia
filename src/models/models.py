import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
import numpy as np
import logging
from torch_geometric.nn import global_add_pool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from mace.calculators import MACECalculator

from ase import Atoms


def compute_E_statistics_vectorized(
    E: np.ndarray, N: np.ndarray, X: np.ndarray, n_species: int,
    delta_vacuum_energies: Dict[int, float], z_map: Dict[int, int]
) -> np.ndarray:
    """Computes per-atom energy regression using a fully vectorized approach."""
    logging.info("Performing vectorized regression to find optimal energy shifts...")

    vacuum_energy_array = np.zeros(n_species)
    for z, energy in delta_vacuum_energies.items():
        if z in z_map:
            idx = z_map[z]
            vacuum_energy_array[idx] = energy
    baseline_energies = X @ vacuum_energy_array
    y = E - baseline_energies

    XTX = X.T @ X
    XTy = X.T @ y
    lam = 1.0
    E_correction = np.linalg.solve(XTX + lam * np.eye(n_species), XTy)
    EperA_regression = vacuum_energy_array + E_correction
    print(EperA_regression)
    logging.info("Regression complete. Final shifts calculated.")
    return EperA_regression


# models.py (updated PCALayer)

class PCALayer(nn.Module):
    """
    A non-trainable PCA layer that dynamically determines the number of components
    needed to retain a certain amount of variance.
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
        
        return x_standardized

    def __repr__(self):
        return (f"PCALayer(in_features={self.in_features}, "
                f"threshold={self.explained_variance_threshold}, "
                f"n_components={self.n_components}, fitted={self.fitted})")

class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    The input is split in half along the last dimension. One half is passed through
    a SiLU activation (the "gate"), which then modulates the other half.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape (..., 2 * dim)
        x_main, x_gate = x.chunk(2, dim=-1)
        return F.silu(x_gate) * x_main



# In models.py

class MLPReadout(nn.Module):
    """
    A flexible Multi-Layer Perceptron for non-linear readout.
    Supports SiLU and SwiGLU activation functions.

    Args:
        in_features (int): Number of input features.
        hidden_features (List[int]): A list where each element is the number
            of neurons in a hidden layer.
        out_features (int): Number of output features. Defaults to 1.
        activation (str): The activation function to use, either 'silu' or 'swiglu'.
            Defaults to 'silu'.
        dropout_p (float): Dropout probability. Defaults to 0.0 (no dropout).
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        out_features: int = 1,
        activation: str = "silu",
        dropout_p: float = 0.0  # <-- NEW ARGUMENT
    ):
        super().__init__()
        
        activation = activation.lower()
        if activation not in ["silu", "swiglu"]:
            raise ValueError("Activation function must be 'silu' or 'swiglu'.")

        self.dropout_p = dropout_p  # <-- NEW

        layers = []
        current_features = in_features

        # Create hidden layers
        for h_features in hidden_features:
            if activation == "silu":
                # Standard SiLU activation
                layers.append(nn.Linear(current_features, h_features, bias=False))
                layers.append(nn.SiLU())
                current_features = h_features
            elif activation == "swiglu":
                # For SwiGLU, the linear layer must output twice the features,
                # as it will be split into a gate and a main branch.
                layers.append(nn.Linear(current_features, 2 * h_features, bias=False))
                layers.append(SwiGLU())
                current_features = h_features # The output dimension of SwiGLU is h_features
            

            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(p=self.dropout_p))


        # Create the final output layer
        layers.append(nn.Linear(current_features, out_features, bias=False)) 

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class ScaleShiftBlock(torch.nn.Module):
    """
    Applies a scale and shift operation.
    Used here to normalize the output of the MLP by the training set std dev.
    """
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [N_atoms, 1]
        # Apply the single-float scale and shift.
        return self.scale * x + self.shift

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale.item():.4f}, shift={self.shift.item():.4f})"



class DualReadoutMACE(nn.Module):
    """
    Wraps a base MACE model to add a trainable (MLP) or fittable (Kernel)
    readout head. This model is used for end-to-end training or for
    the initial feature extraction.
    
    The base MACE model parameters are frozen.
    """
    def __init__(
        self,
        base_mace_model: nn.Module,
        atomic_energy_shifts: torch.Tensor = None,   
        shift_type: str = 'atomic', 
        molecular_energy_shift: float = None,        
        n_atoms_per_molecule: int = None,             
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        mlp_hidden_features: Union[List[int], str] = "auto",
        mlp_activation: str = "silu",
        mlp_dropout_p: float = 0.0, # <-- ADDED THIS ARGUMENT
        use_pca: bool = False,
        pca_variance_threshold: float = 0.99
    ):
        super().__init__()
        self.features = None # This will be populated by the hook
        self.mace_model = base_mace_model
        self.use_pca = use_pca
        self.shift_type = shift_type.lower()
        
        self.molecular_energy_shift = molecular_energy_shift
        self.n_atoms_per_molecule = n_atoms_per_molecule
        
        if self.shift_type == 'molecular' and (self.molecular_energy_shift is None or self.n_atoms_per_molecule is None):
            raise ValueError(
                "For shift_type='molecular' you must provide 'molecular_energy_shift' (float) and 'n_atoms_per_molecule' (int)."
            )
        if self.shift_type == 'molecular' and self.n_atoms_per_molecule <= 0:
             raise ValueError("'n_atoms_per_molecule' must be a positive integer")

        self.mlp_hidden_features_config = mlp_hidden_features
        self.mlp_activation = mlp_activation
        self.mlp_dropout_p = mlp_dropout_p # <-- STORED THIS
    
        logging.info("Freezing parameters of the entire base MACE model...")
        for param in self.mace_model.parameters():
            param.requires_grad = False

        # Find the feature dimension from the MACE model's readout
        if not hasattr(self.mace_model, 'readouts') or not self.mace_model.readouts:
             raise AttributeError("Base MACE model does not have a 'readouts' list.")
        if not hasattr(self.mace_model.readouts[0], 'linear'):
             raise AttributeError("Base MACE model's readout does not have a 'linear' attribute.")
             
        # This assumes the features we want are the *input* to the final linear layer
        num_features = self.mace_model.readouts[0].linear.irreps_in.dim
        logging.info(f"Detected {num_features} input features for the readout head.")
    
        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=num_features,
                explained_variance_threshold=pca_variance_threshold
            )
            logging.info(f"PCA layer enabled with a {pca_variance_threshold:.2%} variance threshold.")

        # The readout head is "lazy" - it will be created in finalize_model()
        self.delta_readout = None
        
        if atomic_energy_shifts is not None:
            self.register_buffer('atomic_energy_shifts', atomic_energy_shifts)
            logging.info("Registered atomic (regression) energy shifts.")
        else:
            self.atomic_energy_shifts = 0 # Use a scalar 0 if not provided

        if self.shift_type == 'atomic' and not isinstance(self.atomic_energy_shifts, torch.Tensor):
            logging.warning("shift_type is 'atomic' but no atomic_energy_shifts tensor was provided.")
            
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

        # Register the hook to capture features *before* the final linear layer
        if not hasattr(self.mace_model, 'products') or not self.mace_model.products:
             raise AttributeError("Base MACE model does not have a 'products' list.")
             
        self.mace_model.products[-1].register_forward_hook(self._hook_fn)
    
    def finalize_model(self, features: torch.Tensor):
        """
        Initializes the readout head (PCA and MLP) after seeing a
        batch of real data.
        
        Args:
            features (Tensor): A tensor of [N_atoms, N_features] from the
                               MACE model hook, used to fit PCA.
        """
        readout_in_features = features.shape[1]
        
        if self.use_pca:
            logging.info(f"Fitting PCA layer to {features.shape[0]} feature vectors...")
            self.pca_layer.fit(features)
            readout_in_features = self.pca_layer.n_components
            
        if isinstance(self.mlp_hidden_features_config, str) and self.mlp_hidden_features_config.lower() == 'auto':
            # Auto-determine a simple MLP structure
            h1_features = readout_in_features
            final_mlp_hidden_features = [h1_features]
            logging.info(f"Automatically determined MLP hidden features: {final_mlp_hidden_features}")
        elif isinstance(self.mlp_hidden_features_config, list):
            final_mlp_hidden_features = self.mlp_hidden_features_config
        else:
            raise TypeError("mlp_hidden_features must be a list of integers or the string 'auto'.")

        logging.info(f"Finalizing model. Readout MLP will have {readout_in_features} input features.")
        self.delta_readout = MLPReadout(
            in_features=readout_in_features,
            hidden_features=final_mlp_hidden_features,
            activation=self.mlp_activation,
            out_features=1,
            dropout_p=self.mlp_dropout_p  # <-- PASS DROPOUT HERE
        )
        
        # Move new modules to the same device/dtype as the base model
        device = next(self.mace_model.parameters()).device
        dtype = next(self.mace_model.parameters()).dtype
        self.delta_readout.to(device=device, dtype=dtype)
        
        if self.scale_shift:
             self.scale_shift.to(device=device, dtype=dtype)
        if self.pca_layer:
             self.pca_layer.to(device=device, dtype=dtype)
 
        logging.info("Model has been finalized and is ready for training.")

    def to(self, *args, **kwargs):
        """ Ensure all sub-modules are moved to the correct device/dtype. """
        super().to(*args, **kwargs)
        # Get device/dtype from the 'to' call itself or from the base model
        device, dtype = None, None
        if 'device' in kwargs:
            device = kwargs['device']
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']

        if device is None:
            device = next(self.mace_model.parameters()).device
        if dtype is None:
            dtype = next(self.mace_model.parameters()).dtype

        if self.pca_layer:
            self.pca_layer.to(device=device, dtype=dtype)
        if self.delta_readout is not None:
            self.delta_readout.to(device=device, dtype=dtype)
        if self.scale_shift is not None:
             self.scale_shift.to(device=device, dtype=dtype)
        if isinstance(self.atomic_energy_shifts, torch.Tensor):
             self.atomic_energy_shifts = self.atomic_energy_shifts.to(device=device, dtype=dtype)
        return self

    def _hook_fn(self, module, input_data, output_data):
        """ Hook to capture the features before the final MACE readout. """
        self.features = output_data

    def forward(self, data: Dict[str, torch.Tensor], compute_force: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the complete model.
        
        Args:
            data (dict): A dictionary of tensors (e.g., from PyG Batch).
                         Must contain 'positions', 'batch', 'node_attrs'.
            compute_force (bool): If True, compute forces via auto-grad.
            
        Returns:
            dict: {"energy": Tensor, "delta_energy": Tensor, "forces": Tensor (optional)}
        """
        if self.delta_readout is None:
            raise RuntimeError(
                "Model is not finalized. Call `model.finalize_model(features)` before the first forward pass."
            )
        
        if compute_force:
            if "positions" not in data:
                 raise ValueError("Cannot compute force: 'positions' not in data.")
            data["positions"].requires_grad_(True)

        # 1. Run the base MACE model
        # This computes the base energy and populates self.features via the hook
        base_output = self.mace_model(data, compute_force=False) # We compute forces manually
        base_energy = base_output["energy"]

        if self.features is None:
            raise RuntimeError("Hook did not capture atomic features. This may be a MACE version issue.")
        
        # 2. Process features (PCA)
        processed_features = self.features
        if self.use_pca:
            processed_features = self.pca_layer(processed_features)

        # 3. Get raw prediction from MLP
        atomic_delta_prediction = self.delta_readout(processed_features)
        
        # 4. Apply scale/shift (the "residual" part)
        scaled_atomic_delta = self.scale_shift(atomic_delta_prediction)

        # 5. Apply 'atomic' shift (if active)
        if self.shift_type == 'atomic':
            if isinstance(self.atomic_energy_shifts, torch.Tensor):
                node_indices = torch.argmax(data['node_attrs'], dim=1)
                shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
                total_atomic_delta = scaled_atomic_delta + shifts.to(scaled_atomic_delta.dtype)
            else:
                total_atomic_delta = scaled_atomic_delta # No shift buffer
        else:
            # For 'molecular' and 'none', the per-atom delta is just the scaled prediction
            total_atomic_delta = scaled_atomic_delta

        # 6. Sum per-atom deltas to get per-graph delta
        batch_map = data["batch"]
        delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)

        # 7. Reconstruct intermediate energy
        final_energy = base_energy + delta_energy
        
        # 8. Apply 'molecular' shift (if active)
        if self.shift_type == 'molecular':
            n_atoms_per_system = torch.bincount(batch_map).to(
                dtype=final_energy.dtype, device=final_energy.device
            )
            # Ensure safe division
            if self.n_atoms_per_molecule == 0:
                raise ValueError("n_atoms_per_molecule cannot be zero.")
            n_molecules = n_atoms_per_system / self.n_atoms_per_molecule
            
            total_molecular_shift = self.molecular_energy_shift * n_molecules
            final_energy = final_energy + total_molecular_shift
            
        elif self.shift_type not in ['atomic', 'none']:
             raise ValueError(f"Invalid shift_type: {self.shift_type}. Must be 'atomic', 'molecular', or 'none'.")

        self.features = None # Clear hook data for the next pass

        output_data = {
            "energy": final_energy,
            "delta_energy": delta_energy
        }

        # 9. Compute forces if requested
        if compute_force:
            forces = -torch.autograd.grad(
                outputs=final_energy.sum(),
                inputs=data["positions"],
                create_graph=self.training, # Create graph if in training mode
                retain_graph=self.training, # Retain graph if in training mode
            )[0]
            output_data["forces"] = forces

        return output_data
