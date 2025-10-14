# models.py
import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def compute_E_statistics_vectorized(
    E: np.ndarray, N: np.ndarray, X: np.ndarray, n_species: int,
    vacuum_energies: Dict[int, float], z_map: Dict[int, int] 
) -> np.ndarray:
    """Computes per-atom energy regression using a fully vectorized approach."""
    logging.info("Performing vectorized regression to find optimal energy shifts...")

    vacuum_energy_array = np.zeros(n_species)
    for z, energy in vacuum_energies.items():
        if z in z_map:
            idx = z_map[z]  # Get the correct 0-based index
            vacuum_energy_array[idx] = energy
    print("vacuum_energy_array ",vacuum_energy_array)
    print("X ",X)

    baseline_energies = X @ vacuum_energy_array
    y = E - baseline_energies

    XTX = X.T @ X
    XTy = X.T @ y

    lam = 1.0  
    E_correction = np.linalg.solve(XTX + lam * np.eye(n_species), XTy)
    
    EperA_regression = vacuum_energy_array + E_correction
    print("EperA_regression ", EperA_regression)
    logging.info("Regression complete. Final shifts calculated.")
    return EperA_regression




class DualReadoutMACE(nn.Module):
    """
    Model version that accounts for a pre-calculated energy shift by
    learning the 'residual' energy.
    """
    def __init__(self, base_mace_model: nn.Module, vacuum_energy_shifts: torch.Tensor = None):
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

        # This head will now learn the residual delta
        self.delta_readout = nn.Linear(num_features, 1, bias=False)
        torch.nn.init.zeros_(self.delta_readout.weight)
        print("Weights of the 'delta_readout' (residual) head have been initialized to zero.")

        # If shifts are provided, register them as a non-trainable buffer.
        # If not, the model will just predict the residual delta energy from the dataset.
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

        # 1. Predict the residual atomic energy (trainable part)
        atomic_delta_prediction = self.delta_readout(self.features)
        
        # 2. If shifts are part of the model, add them
        if isinstance(self.atomic_energy_shifts, torch.Tensor):
            node_indices = torch.argmax(data['node_attrs'], dim=1)
            shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
            total_atomic_delta = atomic_delta_prediction + shifts
        else:
            # If no shifts, the prediction IS the total delta (in this case, the residual)
            total_atomic_delta = atomic_delta_prediction

        # 3. Sum atomic energies to get the total delta energy for the structure
        delta_energy = torch.sum(total_atomic_delta, dim=-2)
        
        self.features = None
        final_energy = base_energy + delta_energy

        output_data = {}
        if compute_force:
            forces = -torch.autograd.grad(
                outputs=final_energy.sum(),
                inputs=data["positions"],
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            output_data["forces"] = forces
            
        output_data["energy"] = final_energy
        return output_data