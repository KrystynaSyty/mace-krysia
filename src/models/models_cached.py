# models_cached.py 
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

from models.models import PCALayer, MLPReadout, ScaleShiftBlock 

class CachedReadoutModel(nn.Module):
    """
    A lightweight model that works on pre-computed features and outputs
    in the same format as DualReadoutMACE.
    """
    def __init__(
        self,
        in_features: int,
        atomic_energy_shifts: torch.Tensor,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        mlp_hidden_features: List[int],    
        mlp_activation: str,              
        use_pca: bool,                    
        pca_variance_threshold: float,    
        z_map: Dict[int, int],
        mlp_dropout_p: float = 0.0,      
        shift_type: str = 'atomic',
        molecular_energy_shift: float = None, # This is now the PER-ATOM shift
        n_atoms_per_molecule: int = None      # No longer used in forward pass
    ):
        super().__init__()
        self.use_pca = use_pca
        self.z_map = z_map
        self.n_species = len(z_map)
        
        self.shift_type = shift_type.lower()
        # self.molecular_energy_shift = molecular_energy_shift # <-- OLD
        self.n_atoms_per_molecule = n_atoms_per_molecule 
        
        self.mlp_dropout_p = mlp_dropout_p
        
        # --- MODIFICATION: Handle Shifts ---
        if self.shift_type == 'atomic':
            if atomic_energy_shifts is not None:
                # Store as a trainable parameter
                self.atomic_energy_shifts = nn.Parameter(atomic_energy_shifts)
                logging.info("Registered 'atomic' shifts as a TRAINABLE nn.Parameter.")
            else:
                logging.warning("shift_type='atomic' but no shifts provided. Using a non-trainable zero buffer.")
                self.register_buffer('atomic_energy_shifts', torch.tensor(0.0, dtype=torch.float64))
        else:
            # Not atomic shift_type, store 0 as a buffer
            self.register_buffer('atomic_energy_shifts', torch.tensor(0.0, dtype=torch.float64))

        if self.shift_type == 'molecular':
            if molecular_energy_shift is not None:
                 # Store as a trainable parameter
                 self.molecular_energy_shift = nn.Parameter(torch.tensor(molecular_energy_shift, dtype=torch.float64))
                 logging.info("Registered 'molecular' shift as a TRAINABLE nn.Parameter.")
            else:
                 raise ValueError("For shift_type='molecular', 'molecular_energy_shift' must be provided.")
        else:
            self.molecular_energy_shift = molecular_energy_shift # Stays None
        # --- END MODIFICATION ---
        
        if self.shift_type == 'molecular' and self.molecular_energy_shift is None:
            # This check is now redundant, but safe to leave
            raise ValueError(
                "For shift_type='molecular' you have to initiate 'molecular_energy_shift' (as the pre-calculated per-atom shift value)."
            )
        self.pca_layer = None
        if self.use_pca:
            self.pca_layer = PCALayer(
                in_features=in_features,
                explained_variance_threshold=pca_variance_threshold
            )

        self.delta_readout = None
        self.mlp_hidden_features_config = mlp_hidden_features
        self.mlp_activation = mlp_activation
        
       # if atomic_energy_shifts is not None:
       #     self.register_buffer('atomic_energy_shifts', atomic_energy_shifts)
       # else:
       #     self.atomic_energy_shifts = 0
            
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def finalize_model(self, features: torch.Tensor):
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
            

        if self.use_pca and self.pca_layer.fitted:
             device = self.pca_layer.mean.device
             dtype = self.pca_layer.mean.dtype
        else:
             device = features.device
             dtype = features.dtype

        logging.info(f"Finalizing model. Readout head will have {readout_in_features} input features.")

        self.delta_readout = MLPReadout(
            in_features=readout_in_features,
            hidden_features=final_mlp_hidden_features,
            activation=self.mlp_activation,
            out_features=1,
            dropout_p=self.mlp_dropout_p  # <-- PASS DROPOUT HERE
        )
        
        self.delta_readout.to(device=device, dtype=dtype)
        
        self.scale_shift.to(device=device, dtype=dtype)
        
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
        
        # 1. Get raw prediction from MLP (dropout is applied here if model is in train() mode)
        atomic_delta_prediction = self.delta_readout(processed_features)
        
        # 2. Apply scale/shift (the "residual" part)
        scaled_atomic_delta = self.scale_shift(atomic_delta_prediction)
        
        # 3. Apply 'atomic' or 'molecular' per-atom shift (if active)
        if self.shift_type == 'atomic':
            if isinstance(self.atomic_energy_shifts, torch.Tensor):
                node_indices = torch.argmax(data['node_attrs'], dim=1)
                shifts = self.atomic_energy_shifts[node_indices].unsqueeze(-1)
                total_atomic_delta = scaled_atomic_delta + shifts
            else:
                total_atomic_delta = scaled_atomic_delta
        
        elif self.shift_type == 'molecular':
            # Apply the single per-atom shift value (pre-calculated and stored) to all atoms
            total_atomic_delta = scaled_atomic_delta + self.molecular_energy_shift
            
        else: # 'none'
            # For 'none', the per-atom delta is just the scaled prediction
            total_atomic_delta = scaled_atomic_delta

        # 4. Sum per-atom deltas to get per-graph delta
        delta_energy = global_add_pool(total_atomic_delta, batch_map).squeeze(-1)
        
        # 5. Reconstruct intermediate energy
        final_energy = base_energy + delta_energy
            
        if self.shift_type not in ['atomic', 'molecular', 'none']:
             raise ValueError(f"Invalid shift_type: {self.shift_type}. Must be 'atomic', 'molecular', or 'none'.")

        return {
            "energy": final_energy,
            "delta_energy": delta_energy
        }
