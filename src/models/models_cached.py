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
from models.models import DualReadoutMACE, compute_E_statistics_vectorized, PCALayer, MLPReadout 


class CachedReadoutModel(nn.Module):
    """
    A lightweight model that works on pre-computed features and outputs
    in the same format as DualReadoutMACE.
    """
    def __init__(
        self,
        in_features: int,
        vacuum_energy_shifts: torch.Tensor,
        mlp_hidden_features: List[int],
        mlp_activation: str,
        use_pca: bool,
        pca_variance_threshold: float,
        z_map: Dict[int, int]
    ):
        super().__init__()
        self.use_pca = use_pca
        self.z_map = z_map
        self.n_species = len(z_map)
        
        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=in_features,
                explained_variance_threshold=pca_variance_threshold
            )

        self.delta_readout = None
        self.mlp_hidden_features_config = mlp_hidden_features
        self.mlp_activation = mlp_activation
        
        if vacuum_energy_shifts is not None:
            self.register_buffer('atomic_energy_shifts', vacuum_energy_shifts)
        else:
            self.atomic_energy_shifts = 0

    def finalize_model(self, features: torch.Tensor):
        """Fits PCA (if used) and initializes the MLP readout head."""
        readout_in_features = features.shape[1]
        
        if self.use_pca:
            logging.info(f"Fitting PCA layer to {features.shape[0]} feature vectors...")
            self.pca_layer.fit(features)
            readout_in_features = self.pca_layer.n_components

        if isinstance(self.mlp_hidden_features_config, str) and self.mlp_hidden_features_config.lower() == 'auto':
            final_mlp_hidden_features = [readout_in_features]
            logging.info(f"Automatically determined MLP hidden features: {final_mlp_hidden_features}")
        else:
            final_mlp_hidden_features = self.mlp_hidden_features_config
        device = self.pca_layer.mean.device
        dtype = self.pca_layer.mean.dtype

        logging.info(f"Finalizing model. Readout head will have {readout_in_features} input features.")
        self.delta_readout = MLPReadout(
            in_features=readout_in_features,
            hidden_features=final_mlp_hidden_features,
            activation=self.mlp_activation,
            out_features=1
        )
        self.delta_readout.to(device=device, dtype=dtype)
        # Zero-initialize the final layer
        final_layer = self.delta_readout.mlp[-1]
        torch.nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
             torch.nn.init.zeros_(final_layer.bias)
        logging.info("CachedReadoutModel has been finalized.")

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes a batch of pre-computed features and graph data, 
        returns the same dict format as the original model.
        """
        if self.delta_readout is None:
            raise RuntimeError("CachedReadoutModel is not finalized. Call .finalize_model(features) first.")
        
        features = data['x']
        batch_map = data['batch_map']
        base_energy = data['base_energy']

        processed_features = self.pca_layer(features) if self.use_pca else features
        atomic_delta_prediction = self.delta_readout(processed_features)
        
        # Apply atomic energy shifts (from regression)
        if isinstance(self.atomic_energy_shifts, torch.Tensor):
            # We need the node_attrs to find the species index
            node_indices = torch.argmax(data['node_attrs'], dim=1)
            shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
            total_atomic_delta = atomic_delta_prediction + shifts
        else:
            total_atomic_delta = atomic_delta_prediction

        # Sum per-atom deltas to get per-graph delta
        delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)
        
        # Reconstruct the final energy (Base MACE Energy + Predicted Delta)
        final_energy = base_energy + delta_energy

        return {
            "energy": final_energy,
            "delta_energy": delta_energy
        }
