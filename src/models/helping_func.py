import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
import numpy as np
import logging
from torch_geometric.nn import global_add_pool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from mace.calculators import MACECalculator

from ase import Atoms

def generate_molecular_shift(molecule, high_accuracy_calculator, low_accuracy_calculator):
    """
    Calculates the energy delta for a single relaxed molecule between two calculators.
    """
    logging.info(f"Relaxing reference molecule {molecule.get_chemical_formula()} with high-accuracy calc...")
    relaxed_molecule = relaxe_molecule(molecule, high_accuracy_calculator)
    relaxed_molecule.calc = high_accuracy_calculator
    E_molecule_high = relaxed_molecule.get_potential_energy()
    logging.info(f"  - High-accuracy energy: {E_molecule_high:.6f} eV")
    relaxed_molecule.calc = low_accuracy_calculator
    E_molecule_low = relaxed_molecule.get_potential_energy()
    logging.info(f"  - Low-accuracy energy: {E_molecule_low:.6f} eV")
    delta_energy = E_molecule_high - E_molecule_low
    logging.info(f"  - Calculated delta (High - Low): {delta_energy:.6f} eV")
    return delta_energy

def get_vacuum_energies(calc_mace_off: MACECalculator, calc_mace_mp: MACECalculator, z_list: List[int]) -> Dict[int, float]:
    """Calculates the energy (mace_off) for single, isolated atoms."""
    logging.info("Calculating vacuum energies for regression baseline...")
    vacuum_energies = {}
    unique_atomic_numbers = sorted(list(set(z_list)))
    
    for z in unique_atomic_numbers:
        atom = Atoms(numbers=[z])
        atom.calc = calc_mace_off
        vacuum_ref = atom.get_potential_energy()
        atom.calc = calc_mace_mp
        vacuum_base = atom.get_potential_energy()
        vacuum_energies[z] = vacuum_ref - vacuum_base
        logging.info(f"  - Referance vacuum energy for Z={z}: {vacuum_ref:.4f} eV")
        logging.info(f"  - Base vacuum energy for Z={z}: {vacuum_base:.4f} eV")
    return vacuum_energies


def print_model_summary(model: nn.Module):
    """
    Prints a detailed summary of the model, including
    total, trainable, and frozen parameters.
    """
    model_name = model.__class__.__name__
    print("\n" + "="*80)
    print(f"           Model Summary: '{model_name}'")
    print("="*80)
    print(model)
    print("-"*80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    print("\nDetails of Trainable Parameters:")
    found_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - Layer: '{name}' | Size: {list(param.shape)} | Status: Trainable")
            found_trainable = True
    
    if not found_trainable:
        print("  - No trainable parameters found.")
        
    print("="*80)


def check_feature_normalization(features: torch.Tensor, tolerance: float = 1e-4):
    """
    Checks if the feature vectors are normalized to length 1 (Unit Norm).
    """
    logging.info("--- Checking Feature Normalization (Post-PCA) ---")
    norms = torch.linalg.norm(features, dim=1)
    mean_norm = torch.mean(norms).item()
    max_norm = torch.max(norms).item()
    min_norm = torch.min(norms).item()
    
    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
    
    logging.info(f"    Mean Norm: {mean_norm:.6f}")
    logging.info(f"    Max Norm:  {max_norm:.6f}")
    logging.info(f"    Min Norm:  {min_norm:.6f}")
    
    if is_normalized:
        logging.info("    [OK] Features are effectively normalized to 1.")
    else:
        logging.warning("    [WARNING] Features are NOT normalized to 1. This may affect Kernel performance.")
    logging.info("-------------------------------------------------")
    return is_normalized

    
