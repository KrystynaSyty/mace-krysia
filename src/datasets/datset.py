import torch
import ase.io
from tqdm import tqdm
import os
from mace.calculators import MACECalculator
from ase import Atoms
from typing import Dict, List
import mbe_automation

def get_vacuum_energies(calc_mace_off: MACECalculator, calc_mace_mp: MACECalculator, z_list: List[int]) -> Dict[int, float]:
    """Calculates the energy (mace_off) for single, isolated atoms."""
    print("Calculating vacuum energies for regression baseline...")
    vacuum_energies = {}
    unique_atomic_numbers = sorted(list(set(z_list)))
    
    for z in unique_atomic_numbers:
        atom = Atoms(numbers=[z])
        atom.calc = calc_mace_off
        vacuum_ref = atom.get_potential_energy()
        atom.calc = calc_mace_mp
        vacuum_base = atom.get_potential_energy()
        vacuum_energies[z] = vacuum_ref - vacuum_base
        print(f"  - Referance vacuum energy for Z={z}: {vacuum_ref:.4f} eV")
        print(f"  - Base vacuum energy for Z={z}: {vacuum_base:.4f} eV")
        
    return vacuum_energies

def process_trajectory(trajectory, calc_mp0, calc_mace_off, vacuum_energy_shifts, description="Processing"):
    """Helper function to process a trajectory and return a list of atoms objects."""
    processed_atoms = []
    for atoms in tqdm(trajectory, desc=description):
        atoms.calc = calc_mp0
        energy_mp0 = atoms.get_potential_energy()
        atoms.calc = calc_mace_off
        energy_mace_off = atoms.get_potential_energy()

        total_delta_energy = energy_mace_off - energy_mp0
        total_vacuum_shift = sum(vacuum_energy_shifts[z] for z in atoms.get_atomic_numbers())
        residual_delta_energy = total_delta_energy - total_vacuum_shift
        
        atoms.info.update({
            'energy_mp0': energy_mp0, 'energy_mace_off': energy_mace_off,
            'total_delta_energy': total_delta_energy, 'residual_delta_energy': residual_delta_energy
        })
        atoms.calc = None
        processed_atoms.append(atoms)
    return processed_atoms
