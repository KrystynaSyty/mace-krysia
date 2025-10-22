# models.py
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
    Model version that accounts for a pre-calculated energy shift by
    learning the 'residual' energy using a non-linear MLP readout.
    """
    def __init__(
        self,
        base_mace_model: nn.Module,
        vacuum_energy_shifts: torch.Tensor = None,
        mlp_hidden_features: List[int] = [32],
        mlp_activation: str = "silu"
    ):
        super().__init__()
        self.features = None
        self.mace_model = base_mace_model

        print("Freezing parameters of the entire base MACE model...")
        for param in self.mace_model.parameters():
            param.requires_grad = False

        if not hasattr(self.mace_model, 'products'):
            raise AttributeError("Base MACE model does not have 'products' attribute. Hook cannot be registered.")

        num_features = self.mace_model.readouts[0].linear.irreps_in.dim
        print(f"Detected {num_features} input features for the readout head.")

        # This head will now learn the residual delta using a non-linear MLP
        print(f"Initializing non-linear MLP readout with activation '{mlp_activation}' and hidden features: {mlp_hidden_features}")
        self.delta_readout = MLPReadout(
            in_features=num_features,
            hidden_features=mlp_hidden_features,
            activation=mlp_activation,
            out_features=1
        )
        
        # Initialize the final layer of the MLP to output zeros initially
        final_layer = self.delta_readout.mlp[-1]
        torch.nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
             torch.nn.init.zeros_(final_layer.bias)
        print("Weights and bias of the final layer of 'delta_readout' MLP have been initialized to zero.")

        if vacuum_energy_shifts is not None:
            self.register_buffer('atomic_energy_shifts', vacuum_energy_shifts)
            print("Vacuum energy shifts have been registered as a model buffer.")
        else:
            self.atomic_energy_shifts = 0
            print("No vacuum energy shifts provided; model will predict the residual delta directly.")

        feature_extractor_layer = self.mace_model.products[-1]
        feature_extractor_layer.register_forward_hook(self._hook_fn)
        print("Hook correctly registered on the last 'product' block.")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dtype = next(self.mace_model.parameters()).dtype
        device = next(self.mace_model.parameters()).device
        self.delta_readout.to(device=device, dtype=dtype)
        return self

    def _hook_fn(self, module, input_data, output_data):
        self.features = output_data

    def forward(self, data: Dict[str, torch.Tensor], compute_force: bool = False) -> Dict[str, torch.Tensor]:
        if compute_force:
            data["positions"].requires_grad_(True)

        base_output = self.mace_model(data, compute_force=False)
        base_energy = base_output["energy"]

        if self.features is None:
            raise RuntimeError("Hook did not capture atomic features.")

        atomic_delta_prediction = self.delta_readout(self.features)

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
