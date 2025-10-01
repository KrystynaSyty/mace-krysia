import torch
import ase
import ase.neighborlist
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from mace.tools import get_atomic_number_table_from_zs, AtomsToData
from typing import Dict

# Importuj klasę modelu z pliku models.py
from models import DualReadoutMACE

class DualMaceASECalculator(Calculator):
    """
    Wrapper dla modelu DualReadoutMACE, który jest zgodny z API kalkulatorów ASE.
    Oblicza energię i siły.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, model: DualReadoutMACE, device: str = 'cpu', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        z_table = get_atomic_number_table_from_zs(
            z for z in self.model.mace_model.atomic_numbers
        )
        self.converter = AtomsToData(
            r_max=self.model.mace_model.r_max.item(),
            z_table=z_table
        )

    def calculate(self, atoms: ase.Atoms = None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        data = self.converter(self.atoms).to(self.device)
        data_dict = data.to_dict()

        should_compute_force = 'forces' in properties
        
        results = self.model(data_dict, compute_force=should_compute_force)

        self.results['energy'] = results['energy'].cpu().numpy().item()
        if should_compute_force:
            self.results['forces'] = results['forces'].cpu().numpy()
