# Calculator.py (updated for lazy initialization model)
import torch
import numpy as np
from typing import Dict, List, Union
from ase.calculators.calculator import Calculator, all_changes
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
# Import MLPReadout to manually reconstruct it
from models import DualReadoutMACE, MLPReadout 


class DualReadoutMACECalculator(Calculator):
    """
    Custom MACE ASE Calculator for the DualReadoutMACE model.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_model_path: str,
        model_path: str,
        # --- MLP configuration is now REQUIRED for reconstruction ---
        mlp_hidden_features: List[int],
        mlp_activation: str,
        device: Union[str, torch.device] = "cpu",
        default_dtype="float64",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.device = torch_tools.init_device(device) if isinstance(device, str) else device
        torch_tools.set_default_dtype(default_dtype)

        print(f"INFO: Loading state dictionary from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)

        print(f"INFO: Loading baseline MACE model from: {base_model_path}")
        base_mace_model = torch.load(f=base_model_path, map_location=self.device)
        
        # --- Step 1: Create the unfinalized model shell ---
        # Infer if PCA was used by checking for 'pca_layer' keys in the saved state.
        use_pca = any(key.startswith("pca_layer") for key in state_dict.keys())
        
        self.model = DualReadoutMACE(
            base_mace_model=base_mace_model,
            vacuum_energy_shifts=state_dict.get('atomic_energy_shifts'),
            mlp_hidden_features=mlp_hidden_features,
            mlp_activation=mlp_activation,
            use_pca=use_pca,
            # pca_variance_threshold is not needed for inference
        )

        # --- Step 2: Manually reconstruct the finalized parts BEFORE loading state_dict ---
        try:
            # Infer the required input dimension for the MLP from the saved weights
            first_mlp_weight_key = 'delta_readout.mlp.0.weight'
            readout_in_features = state_dict[first_mlp_weight_key].shape[1]
            print(f"Reconstructing readout head. Inferred input features: {readout_in_features}")

            # Manually create the delta_readout layer with the correct size
            self.model.delta_readout = MLPReadout(
                in_features=readout_in_features,
                hidden_features=mlp_hidden_features,
                activation=mlp_activation,
                out_features=1
            )
            
            # If PCA was used, set n_components for an accurate model representation
            if use_pca:
                self.model.pca_layer.n_components = readout_in_features
                
        except KeyError:
            raise RuntimeError(f"Could not find '{first_mlp_weight_key}' in the model state_dict. The model file may be incompatible.")

        # --- Step 3: Now load the state dictionary into the correctly-sized model ---
        print("INFO: Loading trained weights into the reconstructed model architecture...")
        self.model.load_state_dict(state_dict)

        self.model.to(device=self.device, dtype=torch.get_default_dtype())
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Calculator initialized successfully on device '{self.device}'.")

        self.r_max = float(base_mace_model.r_max)
        self.z_table = utils.AtomicNumberTable([int(z) for z in base_mace_model.atomic_numbers])

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        batch = self._atoms_to_batch(atoms)
        out = self.model(batch.to_dict(), compute_force="forces" in properties)
        
        if "energy" in properties:
            self.results["energy"] = out["energy"].detach().cpu().item()
        if "forces" in properties:
            self.results["forces"] = out["forces"].detach().cpu().numpy()

    def _atoms_to_batch(self, atoms) -> Dict[str, torch.Tensor]:
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)],
            batch_size=1, shuffle=False, drop_last=False,
        )
        return next(iter(data_loader)).to(self.device)
