import torch
import ase.io
from tqdm import tqdm
import os
import h5py
from mace.calculators import MACECalculator
from ase import Atoms
from ase.io import read
from typing import Dict, List
import mbe_automation.storage as storage
from mbe_automation import FiniteSubsystem, Trajectory, Structure
from dataset.dataset import get_vacuum_energies, process_trajectory
from mbe_automation.calculators.dftb import DFTB3_D4
import math

if __name__ == '__main__':
    # --- CONFIGURATION ---
    base_model_path = "/mnt/storage_3/home/krystyna_syty/models/michaelides_2025/21_urea/MACE_model_swa.model"
    INPUT_HDF5_FILE = 'md_structures.hdf5'
    molecule_path = '/mnt/storage_3/home/krystyna_syty/pl0415-02/project_data/Krysia/mbe-automation/Systems/X23/21_urea/molecule.xyz'
    
    # --- OUTPUT FOLDER CONFIGURATION ---
    OUTPUT_DIRECTORY = 'generated_datasets'
    TRAINING_FILENAME = 'subsample_md_finite_training_biased.xyz' 
    VALIDATION_FILENAME = 'subsample_md_pbc_validation.xyz'
    MOLECULE_FILENAME = 'subsample_md_molecule_gas.xyz' # New output file

    # --- SUBSAMPLING CONFIGURATION (TOTAL BUDGETS) ---
    TOTAL_TRAIN_STRUCTURES = 0
    TOTAL_VALID_STRUCTURES = 0
    TOTAL_MOLECULE_STRUCTURES = 1000 # New budget for molecule
    
    N_CLUSTER_SIZES_TO_SCAN = 8 
    
    # --- HDF5 KEYS ---
    # UPDATED: Points to the 'trajectory' subgroup which contains the 'periodic' attribute.
    # Note: 'molecule[dyn:T=300.00]' assumes this specific temperature group exists.
    MOLECULE_HDF5_KEY = 'training/md_molecule/molecule[dyn:T=300.00]/trajectory'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 0: Setup Output Directory ---
    if not os.path.exists(OUTPUT_DIRECTORY):
        print(f"Creating output directory: {OUTPUT_DIRECTORY}")
        os.makedirs(OUTPUT_DIRECTORY)
    
    training_output_path = os.path.join(OUTPUT_DIRECTORY, TRAINING_FILENAME)
    validation_output_path = os.path.join(OUTPUT_DIRECTORY, VALIDATION_FILENAME)
    molecule_output_path = os.path.join(OUTPUT_DIRECTORY, MOLECULE_FILENAME)

    for path in [base_model_path, INPUT_HDF5_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find required file: '{path}'")

    # --- Step 1: Initialize Calculators ---
    print("Initializing calculators...")
    reference_molecule = read(molecule_path)
    calc_mace_off =  DFTB3_D4(reference_molecule.get_chemical_symbols())

    calc_mp0 = MACECalculator(model_paths=base_model_path, device=device, default_dtype="float64")

    # --- Step 2: Discover Groups (Pressures) ---
    print(f"Inspecting '{INPUT_HDF5_FILE}' for MD pressure groups...")
    md_pressure_groups = []
    
    with h5py.File(INPUT_HDF5_FILE, 'r') as f:
        if 'training' in f and 'md' in f['training']:
            # Filter for keys starting with 'crystal' as per original logic
            md_pressure_groups = [k for k in f['training']['md'].keys() if k.startswith('crystal')]
        else:
            found_keys = list(f.keys())
            raise KeyError(f"Group 'training/md' not found. Top level keys are: {found_keys}")
    
    n_pressure_groups = len(md_pressure_groups)
    print(f"Found {n_pressure_groups} pressure groups: {md_pressure_groups}")
    
    # --- Step 3: Calculate Sampling Rates ---
    print("\n--- Calculating Sampling Rates ---")
    
    # A. Training (Finite Systems)
    total_finite_bins = n_pressure_groups * N_CLUSTER_SIZES_TO_SCAN
    if TOTAL_TRAIN_STRUCTURES > 0:
        samples_per_finite = (TOTAL_TRAIN_STRUCTURES + total_finite_bins - 1) // total_finite_bins
    else:
        samples_per_finite = 0
    print(f"Finite Training Budget: {TOTAL_TRAIN_STRUCTURES} -> Samples per subsystem: {samples_per_finite}")

    # B. Validation (PBC)
    if TOTAL_VALID_STRUCTURES > 0:
        samples_per_pbc = (TOTAL_VALID_STRUCTURES + n_pressure_groups - 1) // n_pressure_groups
    else:
        samples_per_pbc = 0
    print(f"PBC Validation Budget:  {TOTAL_VALID_STRUCTURES} -> Samples per trajectory: {samples_per_pbc}")

    # C. Molecule (Gas Phase)
    # Since all molecule structures are under one key, the rate is simply the total budget.
    samples_per_molecule = TOTAL_MOLECULE_STRUCTURES
    print(f"Molecule Gas Budget:    {TOTAL_MOLECULE_STRUCTURES} -> Samples total: {samples_per_molecule}")


    # --- Step 4: Generate TRAINING Dataset (Finite Subsystems) ---
    print(f"\n--- Generating TRAINING dataset from Finite Subsystems (MD) ---")
    training_atoms = []

    if samples_per_finite > 0:
        for group_name in tqdm(md_pressure_groups, desc="Processing Pressure Groups (Train)"):
            finite_base_key = f"training/md/{group_name}/finite"
            
            for n in range(1, N_CLUSTER_SIZES_TO_SCAN + 1): 
                key = f"{finite_base_key}/n={n}"
                try:
                    subsystem_sampled = FiniteSubsystem.read(dataset=INPUT_HDF5_FILE, key=key).subsample(n=samples_per_finite)
                    trajectory = storage.ASETrajectory(subsystem_sampled.cluster_of_molecules)
                    processed_frames = process_trajectory(
                        trajectory, calc_mp0, calc_mace_off,
                        description=f"Train: {group_name} n={n}"
                    )
                    
                    for at in processed_frames:
                        at.info['src_group'] = group_name
                        at.info['cluster_size'] = n

                    training_atoms.extend(processed_frames)
                except KeyError:
                    pass
                except Exception as e:
                    print(f"Error processing '{key}': {e}")
    else:
        print("Skipping Training generation (budget is 0).")

    # --- Step 5: Generate VALIDATION Dataset (PBC Trajectories) ---
    print(f"\n--- Generating VALIDATION dataset from PBC Trajectories (MD) ---")
    validation_atoms = []

    if samples_per_pbc > 0:
        for group_name in tqdm(md_pressure_groups, desc="Processing Pressure Groups (Valid)"):
            pbc_key = f"training/md/{group_name}/trajectory"
            try:
                md_trajectory_full = Trajectory.read(dataset=INPUT_HDF5_FILE, key=pbc_key)
                md_trajectory_sampled = md_trajectory_full.subsample(n=samples_per_pbc)
                md_trajectory_ase = storage.ASETrajectory(md_trajectory_sampled)
                
                processed_frames = process_trajectory(
                    md_trajectory_ase, calc_mp0, calc_mace_off,
                    description=f"Valid: {group_name} PBC"
                )
                
                for at in processed_frames:
                    at.info['src_group'] = group_name
                    at.info['type'] = 'pbc_validation'

                validation_atoms.extend(processed_frames)
            except (KeyError, AttributeError) as e:
                print(f"Warning: Could not load MD trajectory from key '{pbc_key}'. Error: {e}. Skipping.")
    else:
        print("Skipping Validation generation (budget is 0).")

    # --- Step 6: Generate MOLECULE Dataset (Gas Phase) ---
    print(f"\n--- Generating MOLECULE dataset (Gas Phase) ---")
    molecule_atoms = []
    
    if samples_per_molecule > 0:
        try:
            print(f"Reading molecule data from key: {MOLECULE_HDF5_KEY}")
            
            # Attempt to read as Trajectory first
            try:
                mol_trajectory_full = Trajectory.read(dataset=INPUT_HDF5_FILE, key=MOLECULE_HDF5_KEY)
            except Exception:
                # Fallback: Try reading as Structure if Trajectory fails
                print("Trajectory read failed, trying to read as Structure...")
                mol_trajectory_full = Structure.read(dataset=INPUT_HDF5_FILE, key=MOLECULE_HDF5_KEY)

            # --- FIX: Compute feature vectors before subsampling ---
            print("Computing feature vectors for molecule subsampling...")
            # We must run the model to generate 'averaged_environments' descriptors 
            # so that subsample() can calculate distances.
            mol_trajectory_full.run_model(
                calculator=calc_mp0,
                energies=False,
                forces=False,
                feature_vectors_type="averaged_environments"
            )

            # Subsample
            mol_sampled = mol_trajectory_full.subsample(n=samples_per_molecule)
            
            # Convert to ASE (Logic mirrors Step 5)
            mol_ase = storage.ASETrajectory(mol_sampled)

            processed_frames = process_trajectory(
                mol_ase, calc_mp0, calc_mace_off,
                description="Molecule Gas Phase"
            )

            for at in processed_frames:
                at.info['src_group'] = 'molecule_gas'
                at.info['type'] = 'molecule'

            molecule_atoms.extend(processed_frames)

        except (KeyError, AttributeError, Exception) as e:
             print(f"Warning: Could not load Molecule data from key '{MOLECULE_HDF5_KEY}'. Error: {e}. Skipping.")
    else:
        print("Skipping Molecule generation (budget is 0).")


    # --- Step 7: Save the Final Datasets ---
    if training_atoms:
        print(f"\nTotal frames collected for TRAINING: {len(training_atoms)}")
        print(f"Saving TRAINING dataset to: '{training_output_path}'")
        ase.io.write(training_output_path, training_atoms)
    
    if validation_atoms:
        print(f"\nTotal frames collected for VALIDATION: {len(validation_atoms)}")
        print(f"Saving VALIDATION dataset to: '{validation_output_path}'")
        ase.io.write(validation_output_path, validation_atoms)

    if molecule_atoms:
        print(f"\nTotal frames collected for MOLECULE: {len(molecule_atoms)}")
        print(f"Saving MOLECULE dataset to: '{molecule_output_path}'")
        ase.io.write(molecule_output_path, molecule_atoms)

    print("\nDone! The datasets have been created successfully.")
