import torch
import ase.io
from tqdm import tqdm
import os
from mace.calculators import MACECalculator
from ase import Atoms
from typing import Dict, List
import mbe_automation
import numpy as np
from mbe_automation import Structure
import mbe_automation.storage  # Needed for conversion   

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



def ase_list_to_structure(atoms_list):
    """
    Manually convert a list of ASE Atoms objects into an mbe_automation.Structure object.
    This handles stacking positions, cells, energies, and forces into (N_frames, ...) arrays.
    """
    if not atoms_list:
        raise ValueError("Empty atoms list provided.")

    # Assume topology (atomic numbers, masses) is constant across the trajectory
    first_frame = atoms_list[0]
    n_frames = len(atoms_list)
    n_atoms = len(first_frame)

    # 1. Stack Geometry
    # Positions: shape (n_frames, n_atoms, 3)
    positions = np.array([frame.get_positions() for frame in atoms_list])
    
    # Cell Vectors: shape (n_frames, 3, 3)
    cell_vectors = np.array([frame.get_cell() for frame in atoms_list])

    # Atomic Numbers & Masses (1D arrays)
    atomic_numbers = first_frame.get_atomic_numbers()
    masses = first_frame.get_masses()

    # 2. Stack Energies (if available)
    # Check info['energy'], info['free_energy'], or calculator
    E_pot_list = []
    has_energy = True
    for frame in atoms_list:
        # Try standard ASE extended XYZ keys
        e = frame.info.get('energy') or frame.info.get('free_energy')
        if e is None:
            has_energy = False
            break
        E_pot_list.append(e)
    
    E_pot = np.array(E_pot_list) if has_energy else None

    # 3. Stack Forces (if available)
    # Usually stored in atoms.arrays['forces']
    forces_list = []
    has_forces = True
    for frame in atoms_list:
        f = frame.arrays.get('forces')
        if f is None:
            has_forces = False
            break
        forces_list.append(f)
        
    forces = np.array(forces_list) if has_forces else None

    # 4. Construct Structure
    return Structure(
        positions=positions,
        atomic_numbers=atomic_numbers,
        masses=masses,
        cell_vectors=cell_vectors,
        n_frames=n_frames,
        n_atoms=n_atoms,
        E_pot=E_pot,
        forces=forces,
        feature_vectors=None,
        feature_vectors_type="none",
        delta=None
    )

