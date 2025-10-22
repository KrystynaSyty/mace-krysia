import torch
import numpy as np
from typing import Dict, Optional, Union
from typing import Dict, List
from ase.calculators.calculator import Calculator, all_changes
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from models import DualReadoutMACE

class DualReadoutMACECalculator(Calculator):
    """
    Custom MACE ASE Calculator for the DualReadoutMACE model.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_model_path: str,
        model_path: str,
        vacuum_energy_shifts: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        default_dtype="float64",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        if isinstance(device, str):
            self.device = torch_tools.init_device(device)
        else:
            self.device = device

        torch_tools.set_default_dtype(default_dtype)

        # 1. Load the state dictionary from your trained DualReadoutMACE model.
        #    This file contains the learned weights and any saved buffers (like atomic_energy_shifts).
        print(f"INFO: Loading state dictionary from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)

        # 2. Extract the 'atomic_energy_shifts' buffer from the state_dict if it exists.
        #    This is essential for recreating the exact model architecture before loading weights.
        #    We use state_dict.get() to safely return None if the key doesn't exist.
        extracted_shifts = state_dict.get('atomic_energy_shifts', None)
        if extracted_shifts is not None:
            print("INFO: Found and extracted 'atomic_energy_shifts' from the model file.")
        else:
            print("INFO: No 'atomic_energy_shifts' buffer found in the model file.")

        # 3. Load the base MACE model. This is a required component to build the
        #    DualReadoutMACE architecture.
        print(f"INFO: Loading baseline MACE model from: {base_model_path}")
        base_mace_model = torch.load(f=base_model_path, map_location=self.device)
        
        # 4. Now, create an instance of the DualReadoutMACE model architecture.
        #    It's critical to use the base model and the extracted shifts to ensure
        #    the architecture is identical to the one that was saved.
        self.model = DualReadoutMACE(
            base_mace_model=base_mace_model,
            vacuum_energy_shifts=extracted_shifts # Pass the extracted shifts here
        )

        # 5. Finally, load the weights and buffers from the state_dict into the
        #    correctly-built model instance.
        print("INFO: Loading trained weights into the model architecture...")
        self.model.load_state_dict(state_dict)

        if not isinstance(self.model, DualReadoutMACE):
            raise TypeError("The loaded model is not a DualReadoutMACE instance.")

        # Common setup for inference
        self.model.to(device=self.device, dtype=torch.get_default_dtype())
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Calculator initialized successfully on device '{self.device}'.")

        # Store necessary parameters for the calculator
        self.r_max = float(base_mace_model.r_max)
        self.z_table = utils.AtomicNumberTable([int(z) for z in base_mace_model.atomic_numbers])

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        batch = self._atoms_to_batch(atoms)
        out = self.model(batch.to_dict(), compute_force=True)
        self.results["energy"] = out["energy"].detach().cpu().item()
        self.results["forces"] = out["forces"].detach().cpu().numpy()

    def _atoms_to_batch(self, atoms) -> Dict[str, torch.Tensor]:
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)],
            batch_size=1, shuffle=False, drop_last=False,
        )
        return next(iter(data_loader)).to(self.device)
