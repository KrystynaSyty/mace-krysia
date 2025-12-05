import torch
import ase.io
from tqdm import tqdm
import os
import h5py
from mace.calculators import MACECalculator
from ase import Atoms
from typing import Dict, List
import mbe_automation.storage as storage 
from dataset.dataset import get_vacuum_energies, process_trajectory

if __name__ == '__main__':
    # --- CONFIGURATION ---
    high_accuracy_model_path = "/mnt/storage_3/home/krystyna_syty/models/mace/MACE-omol-0-extra-large-1024.model"
    base_model_path = "/mnt/storage_3/home/krystyna_syty/models/michaelides_2025/21_urea/MACE_model_swa.model"
    INPUT_HDF5_FILE = 'md_structures.hdf5' 
    
    TRAINING_OUTPUT_FILE = 'subsample_md_finite_training.xyz'
    VALIDATION_OUTPUT_FILE = 'subsample_md_pbc_validation.xyz'

    # --- SUBSAMPLING CONFIGURATION ---
    N_SAMPLES_PER_SYSTEM_TRAIN = 10
    N_SAMPLES_PER_SYSTEM_VALID = 10
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for path in [high_accuracy_model_path, base_model_path, INPUT_HDF5_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find required file: '{path}'")

    # --- Step 1: Initialize Calculators ---
    print("Initializing calculators...")
    calc_mace_off = MACECalculator(model_paths=high_accuracy_model_path, device=device, default_dtype="float64")
    calc_mp0 = MACECalculator(model_paths=base_model_path, device=device, default_dtype="float64")

    # --- Step 2: Discover Groups (Pressures) ---
    print(f"Inspecting '{INPUT_HDF5_FILE}' for MD pressure groups...")
    md_pressure_groups = []
    
    # UPDATED LOGIC HERE: Look inside 'training/md'
    with h5py.File(INPUT_HDF5_FILE, 'r') as f:
        if 'training' in f and 'md' in f['training']:
            # Get all keys under 'training/md' that start with 'crystal'
            md_pressure_groups = [k for k in f['training']['md'].keys() if k.startswith('crystal')]
        else:
            # Helper error message to see what is actually there if it fails again
            found_keys = list(f.keys())
            raise KeyError(f"Group 'training/md' not found. Top level keys are: {found_keys}")
    
    print(f"Found {len(md_pressure_groups)} pressure groups: {md_pressure_groups}")

    # --- Step 3: Calculate Vacuum Energy Shifts ---
    print("Reading atomic numbers from a sample subsystem to calculate vacuum shifts...")
    sample_group = md_pressure_groups[0]
    # UPDATED PATH: training/md/...
    temp_subsystem_key = f"training/md/{sample_group}/finite/n=1"
    
    try:
        temp_subsystem = storage.read_finite_subsystem(dataset=INPUT_HDF5_FILE, key=temp_subsystem_key)
        all_zs = temp_subsystem.cluster_of_molecules.atomic_numbers
        vacuum_energy_shifts = get_vacuum_energies(calc_mace_off, calc_mp0, all_zs)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate vacuum shifts using key '{temp_subsystem_key}': {e}")
    
    # --- Step 4: Generate TRAINING Dataset (Finite Subsystems) ---
    print("\n--- Generating TRAINING dataset from Finite Subsystems (MD) ---")
    training_atoms = []

    for group_name in tqdm(md_pressure_groups, desc="Processing Pressure Groups (Train)"):
        # UPDATED PATH: training/md/...
        finite_base_key = f"training/md/{group_name}/finite"
        
        for n in range(1, 9): # n=1 to n=8
            key = f"{finite_base_key}/n={n}"
            
            try:
                subsystem_full = storage.read_finite_subsystem(dataset=INPUT_HDF5_FILE, key=key)
                subsystem_sampled = subsystem_full.subsample(n=N_SAMPLES_PER_SYSTEM_TRAIN)
                
                trajectory = storage.ASETrajectory(subsystem_sampled.cluster_of_molecules)
                processed_frames = process_trajectory(
                    trajectory, calc_mp0, calc_mace_off, vacuum_energy_shifts,
                    description=f"Train: {group_name} n={n}"
                )
                
                for at in processed_frames:
                    at.info['src_group'] = group_name
                    at.info['cluster_size'] = n

                training_atoms.extend(processed_frames)
            except KeyError:
                print(f"Warning: Subsystem not found for key '{key}'. Skipping.")
            except Exception as e:
                print(f"Error processing '{key}': {e}")

    # --- Step 5: Generate VALIDATION Dataset (PBC Trajectories) ---
    print("\n--- Generating VALIDATION dataset from PBC Trajectories (MD) ---")
    validation_atoms = []

    for group_name in tqdm(md_pressure_groups, desc="Processing Pressure Groups (Valid)"):
        # UPDATED PATH: training/md/...
        pbc_key = f"training/md/{group_name}/trajectory"
        
        try:
            md_trajectory_full = storage.read_trajectory(dataset=INPUT_HDF5_FILE, key=pbc_key)
            md_trajectory_sampled = md_trajectory_full.subsample(n=N_SAMPLES_PER_SYSTEM_VALID)
            md_trajectory_ase = storage.ASETrajectory(md_trajectory_sampled)
            
            processed_frames = process_trajectory(
                md_trajectory_ase, calc_mp0, calc_mace_off, vacuum_energy_shifts,
                description=f"Valid: {group_name} PBC"
            )
            
            for at in processed_frames:
                at.info['src_group'] = group_name
                at.info['type'] = 'pbc_validation'

            validation_atoms.extend(processed_frames)
        except (KeyError, AttributeError) as e:
            print(f"Warning: Could not load MD trajectory from key '{pbc_key}'. Error: {e}. Skipping.")

    # --- Step 6: Save the Final Datasets ---
    print(f"\nTotal frames in TRAINING set: {len(training_atoms)}")
    print(f"Saving TRAINING dataset to file: '{TRAINING_OUTPUT_FILE}'")
    ase.io.write(TRAINING_OUTPUT_FILE, training_atoms)
    
    if validation_atoms:
        print(f"\nTotal frames in VALIDATION set: {len(validation_atoms)}")
        print(f"Saving VALIDATION dataset to file: '{VALIDATION_OUTPUT_FILE}'")
        ase.io.write(VALIDATION_OUTPUT_FILE, validation_atoms)

    print("\nDone! The datasets have been created successfully.")
