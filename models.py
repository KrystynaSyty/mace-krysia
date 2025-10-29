import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
import logging
from torch_geometric.nn import global_add_pool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        out_features: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        
        activation = activation.lower()
        if activation not in ["silu", "swiglu"]:
            raise ValueError("Activation function must be 'silu' or 'swiglu'.")

        layers = []
        current_features = in_features

        # Create hidden layers
        for h_features in hidden_features:
            if activation == "silu":
                # Standard SiLU activation
                layers.append(nn.Linear(current_features, h_features))
                layers.append(nn.SiLU())
                current_features = h_features
            elif activation == "swiglu":
                # For SwiGLU, the linear layer must output twice the features,
                # as it will be split into a gate and a main branch.
                layers.append(nn.Linear(current_features, 2 * h_features))
                layers.append(SwiGLU())
                current_features = h_features # The output dimension of SwiGLU is h_features

        # Create the final output layer
        layers.append(nn.Linear(current_features, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)



class DualReadoutMACE(nn.Module):
    """
    Model with a lazily-initialized readout head, allowing PCA dimensionality
    to be determined dynamically from data.
    """
    def __init__(
        self,
        base_mace_model: nn.Module,
        vacuum_energy_shifts: torch.Tensor = None,
        mlp_hidden_features: List[int] = [32],
        mlp_activation: str = "silu",
        use_pca: bool = False,
        pca_variance_threshold: float = 0.99 # <<< Takes threshold directly
    ):
        super().__init__()
        self.features = None
        self.mace_model = base_mace_model
        self.use_pca = use_pca
        
        # --- Store config for later initialization ---
        self.mlp_hidden_features = mlp_hidden_features
        self.mlp_activation = mlp_activation
        
        # --- Initialize layers that don't depend on data ---
        print("Freezing parameters of the entire base MACE model...")
        for param in self.mace_model.parameters():
            param.requires_grad = False

        if not hasattr(self.mace_model, 'products'):
            raise AttributeError("Base MACE model does not have 'products' attribute.")

        num_features = self.mace_model.readouts[0].linear.irreps_in.dim
        print(f"Detected {num_features} input features for the readout head.")

        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=num_features,
                explained_variance_threshold=pca_variance_threshold
            )
            print(f"PCA layer enabled with a {pca_variance_threshold:.2%} variance threshold.")

        # The readout layer is NOT created yet. It will be created in finalize_model().
        self.delta_readout = None
        
        if vacuum_energy_shifts is not None:
            self.register_buffer('atomic_energy_shifts', vacuum_energy_shifts)
        else:
            self.atomic_energy_shifts = 0

        self.mace_model.products[-1].register_forward_hook(self._hook_fn)

    def finalize_model(self, features: torch.Tensor):
        """
        Finalizes the model architecture by fitting PCA (if used) and creating the readout layer.
        This method MUST be called before training or inference.
        """
        readout_in_features = features.shape[1]
        
        if self.use_pca:
            self.pca_layer.fit(features)
            readout_in_features = self.pca_layer.n_components
        
        print(f"Finalizing model. Readout head will have {readout_in_features} input features.")
        self.delta_readout = MLPReadout(
            in_features=readout_in_features,
            hidden_features=self.mlp_hidden_features,
            activation=self.mlp_activation,
            out_features=1
        )
        
        # Move new layer to the correct device and dtype
        device = next(self.mace_model.parameters()).device
        dtype = next(self.mace_model.parameters()).dtype
        self.delta_readout.to(device=device, dtype=dtype)
        
        # Initialize final layer weights to zero
        final_layer = self.delta_readout.mlp[-1]
        torch.nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
             torch.nn.init.zeros_(final_layer.bias)
        print("Model has been finalized and is ready for training.")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dtype = next(self.mace_model.parameters()).dtype
        device = next(self.mace_model.parameters()).device
        if self.pca_layer:
            self.pca_layer.to(device=device, dtype=dtype)
        self.delta_readout.to(device=device, dtype=dtype)
        return self

    def _hook_fn(self, module, input_data, output_data):
        self.features = output_data

    def forward(self, data: Dict[str, torch.Tensor], compute_force: bool = False) -> Dict[str, torch.Tensor]:
        if self.delta_readout is None:
            raise RuntimeError(
                "Model is not finalized. Call `model.finalize_model(features)` before the first forward pass."
            )
        if compute_force:
            data["positions"].requires_grad_(True)

        base_output = self.mace_model(data, compute_force=False)
        base_energy = base_output["energy"]

        if self.features is None:
            raise RuntimeError("Hook did not capture atomic features.")
        
        processed_features = self.features
        if self.use_pca:
            processed_features = self.pca_layer(processed_features)

        atomic_delta_prediction = self.delta_readout(processed_features)

        if isinstance(self.atomic_energy_shifts, torch.Tensor):
            node_indices = torch.argmax(data['node_attrs'], dim=1)
            shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
            total_atomic_delta = atomic_delta_prediction + shifts
        else:
            total_atomic_delta = atomic_delta_prediction

        batch_map = data["batch"]
        delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)

        self.features = None
        final_energy = base_energy + delta_energy

        output_data = {
            "energy": final_energy,
            "delta_energy": delta_energy
        }

        if compute_force:
            forces = -torch.autograd.grad(
                outputs=final_energy.sum(),
                inputs=data["positions"],
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            output_data["forces"] = forces

        return output_data
