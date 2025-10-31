import torch
import ase.io
from tqdm import tqdm
import os
from mace.calculators import MACECalculator
from ase import Atoms
from typing import Dict, List
import mbe_automation
from dataset.dataset import get_vacuum_energies, process_trajectory

if __name__ == '__main__':
  
    high_accuracy_model_path = "{high_accuracy_model_path.model}"
    base_model_path = "{base_model_path.model}"
    INPUT_HDF5_FILE = '{input_file.hdf5}'
    
    TRAINING_OUTPUT_FILE = '{training.xyz}'
    VALIDATION_OUTPUT_FILE = '{validation.xyz}'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for path in [high_accuracy_model_path, base_model_path, INPUT_HDF5_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find required file: '{path}'")

    # --- Step 1: Initialize Calculators ---
    print("Initializing calculators...")
    calc_mace_off = MACECalculator(model_paths=high_accuracy_model_path, device=device, default_dtype="float64")
    calc_mp0 = MACECalculator(model_paths=base_model_path, device=device, default_dtype="float64")

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
