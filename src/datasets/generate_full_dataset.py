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

if __name__ == '__main__':
    # --- CONFIGURATION ---
    MACE_OFF_MODEL_PATH = 'MACE-OFF24_medium.model'
    MP0_MODEL_PATH = 'MACE-MP_small.model'
    INPUT_HDF5_FILE = 'training_set_08_cyanamide_big_md_sampling.hdf5'
    
    TRAINING_OUTPUT_FILE = 'combined_finite_training_dataset.xyz'
    VALIDATION_OUTPUT_FILE = 'combined_pbc_validation_dataset.xyz'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for path in [MACE_OFF_MODEL_PATH, MP0_MODEL_PATH, INPUT_HDF5_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find required file: '{path}'")

    # --- Step 1: Initialize Calculators ---
    print("Initializing calculators...")
    calc_mace_off = MACECalculator(model_paths=MACE_OFF_MODEL_PATH, device=device, default_dtype="float64")
    calc_mp0 = MACECalculator(model_paths=MP0_MODEL_PATH, device=device, default_dtype="float64")

    # --- Step 2: Calculate Vacuum Energy Shifts ---
    print("Reading atomic numbers from a sample subsystem to calculate vacuum shifts...")
    # Read from a representative subsystem to get all atomic numbers present
    temp_subsystem_key = "training/md_sampling/crystal[dyn:T=298.15,p=0.00010]/finite_subsystems/n=1"
    temp_subsystem = mbe_automation.storage.read_finite_subsystem(dataset=INPUT_HDF5_FILE, key=temp_subsystem_key)
    all_zs = temp_subsystem.cluster_of_molecules.atomic_numbers
    vacuum_energy_shifts = get_vacuum_energies(calc_mace_off, calc_mp0, all_zs)
    
    # --- Step 3: Generate TRAINING Dataset from ALL Finite Subsystems ---
    print("\n--- Generating TRAINING dataset from ALL Finite Subsystems ---")
    training_atoms = []
    
    # Process finite subsystems from md_sampling
    md_subsystem_base_key = "training/md_sampling/crystal[dyn:T=298.15,p=0.00010]/finite_subsystems"
    for i in range(1, 9): # n=1 to n=8
        key = f"{md_subsystem_base_key}/n={i}"
        print(f"\n--- Loading MD finite subsystem from key: {key} ---")
        try:
            subsystem = mbe_automation.storage.read_finite_subsystem(dataset=INPUT_HDF5_FILE, key=key)
            trajectory = mbe_automation.storage.ASETrajectory(subsystem.cluster_of_molecules)
            print(f"Found {len(trajectory)} frames to process for n={i}.")
            processed_frames = process_trajectory(
                trajectory, calc_mp0, calc_mace_off, vacuum_energy_shifts,
                description=f"Processing MD finite subsystem n={i}"
            )
            training_atoms.extend(processed_frames)
        except KeyError:
            print(f"Warning: Subsystem not found for key '{key}'. Skipping.")
    
    # Process finite subsystems from phonon_sampling
    phonon_subsystem_base_key = "training/phonon_sampling/finite_subsystems"
    for i in range(1, 9): # n=1 to n=8
        key = f"{phonon_subsystem_base_key}/n={i}"
        print(f"\n--- Loading Phonon finite subsystem from key: {key} ---")
        try:
            subsystem = mbe_automation.storage.read_finite_subsystem(dataset=INPUT_HDF5_FILE, key=key)
            trajectory = mbe_automation.storage.ASETrajectory(subsystem.cluster_of_molecules)
            print(f"Found {len(trajectory)} frames to process for n={i}.")
            processed_frames = process_trajectory(
                trajectory, calc_mp0, calc_mace_off, vacuum_energy_shifts,
                description=f"Processing Phonon finite subsystem n={i}"
            )
            training_atoms.extend(processed_frames)
        except KeyError:
            print(f"Warning: Subsystem not found for key '{key}'. Skipping.")

    # --- Step 4: Generate VALIDATION Dataset from ALL PBC Trajectories ---
    print("\n--- Generating VALIDATION dataset from ALL PBC Trajectories ---")
    validation_atoms = []
    
    # List of all PBC trajectory keys
    pbc_keys = [
        "training/md_sampling/crystal[dyn:T=298.15,p=0.00010]/trajectory"
        # Add other PBC keys here if they exist. For now, there's only one.
    ]

    for md_key in pbc_keys:
        print(f"\n--- Loading PBC trajectory from key: {md_key} ---")
        try:
            md_trajectory_data = mbe_automation.storage.read_trajectory(dataset=INPUT_HDF5_FILE, key=md_key)
            md_trajectory = mbe_automation.storage.ASETrajectory(md_trajectory_data)
            print(f"Found {len(md_trajectory)} frames to process for validation set.")
            
            processed_frames = process_trajectory(
                md_trajectory, calc_mp0, calc_mace_off, vacuum_energy_shifts,
                description=f"Processing PBC validation trajectory"
            )
            validation_atoms.extend(processed_frames)
        except (KeyError, AttributeError) as e:
            print(f"Warning: Could not load MD trajectory from key '{md_key}'. Error: {e}. Skipping.")

    # --- Step 5: Save the Final Datasets ---
    print(f"\nTotal frames in TRAINING set: {len(training_atoms)}")
    print(f"Saving TRAINING dataset to file: '{TRAINING_OUTPUT_FILE}'")
    ase.io.write(TRAINING_OUTPUT_FILE, training_atoms)
    
    if validation_atoms:
        print(f"\nTotal frames in VALIDATION set: {len(validation_atoms)}")
        print(f"Saving VALIDATION dataset to file: '{VALIDATION_OUTPUT_FILE}'")
        ase.io.write(VALIDATION_OUTPUT_FILE, validation_atoms)

    print("\nDone! The datasets have been created successfully.")
