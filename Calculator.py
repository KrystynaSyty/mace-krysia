###########################################################################################
# An ASE Calculator for the custom DualReadoutMACE model.
#
# This version has been made more flexible to support two use cases:
#  1. Creating a new delta-learning model from a baseline MACE model.
#  2. Loading a fully trained DualReadoutMACE model directly.
###########################################################################################

import torch
import numpy as np
from typing import Dict, Optional

# ASE imports
from ase.calculators.calculator import Calculator, all_changes

# MACE imports - ensure 'mace-torch' is installed
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

# Your custom model import
from models import DualReadoutMACE



class DeltaMaceCalculator(Calculator):
    """
    Custom MACE ASE Calculator for the DualReadoutMACE model.

    This calculator can be initialized in two ways:

    1. To create a NEW delta model for training or inference:
       >> calc = DualReadoutMACECalculator(base_model_path='path/to/base.model')

    2. To load a fully TRAINED delta model for inference:
       >> calc = DualReadoutMACECalculator(model_path='path/to/trained_delta.pt')
    """
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        
        # --- MODIFIED LOGIC ---
        # Ensure exactly one model source is provided to avoid ambiguity.
        num_provided = sum(arg is not None for arg in [base_model_path, model_path, model])
        if num_provided != 1:
            raise ValueError(
                "You must provide exactly one of 'base_model_path' (to create a new model), "
                "'model_path' (to load a trained model), or a 'model' object."
            )

        self.device = torch_tools.init_device(device)
        torch_tools.set_default_dtype(default_dtype)
        
        base_mace_model = None

        if base_model_path is not None:
            # --- Scenario A: Create a NEW delta model from a base model ---
            print(f"Loading baseline MACE model from: {base_model_path}")
            base_mace_model = torch.load(f=base_model_path, map_location=self.device)
            print("Wrapping baseline model with a new DualReadoutMACE...")
            self.model = DualReadoutMACE(base_mace_model=base_mace_model)

        elif model_path is not None:
            # --- Scenario B: Load a pre-trained, complete delta model ---
            print(f"Loading fully trained DualReadoutMACE model from: {model_path}")
            self.model = torch.load(f=model_path, map_location=self.device)
            # The base model is a submodule of the loaded model
            base_mace_model = self.model.mace_model
        
        elif model is not None:
            # --- Scenario C: Use a pre-initialized model object ---
            print("Using a provided DualReadoutMACE model object.")
            self.model = model
            base_mace_model = self.model.mace_model

        if not isinstance(self.model, DualReadoutMACE):
            raise TypeError("The loaded or created model is not a DualReadoutMACE instance.")

        # Set up device, dtype, and freeze parameters for inference
        self.model.to(device=self.device, dtype=torch.get_default_dtype())
        self.model.eval() # Set to evaluation mode

        # Since this is a calculator for inference, we freeze the gradients.
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Calculator initialized on device '{self.device}' with dtype '{default_dtype}'.")

        # Store constants and utilities from the base model
        self.r_max = float(base_mace_model.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in base_mace_model.atomic_numbers]
        )
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        batch = self._atoms_to_batch(atoms)
        out = self.model(batch.to_dict(), compute_force=True)
        self.results["energy"] = out["energy"].detach().cpu().item() * self.energy_units_to_eV
        self.results["forces"] = out["forces"].detach().cpu().numpy() * (self.energy_units_to_eV / self.length_units_to_A)

    def _atoms_to_batch(self, atoms) -> Dict[str, torch.Tensor]:
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)],
            batch_size=1, shuffle=False, drop_last=False,
        )
        return next(iter(data_loader)).to(self.device)


