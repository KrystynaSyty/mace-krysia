import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from typing import List
import ase
import ase.io
from tqdm import tqdm
import os

from models import DualReadoutMACE

def prepare_dataset_from_xyz(file_path: str) -> List[ase.Atoms]:
    """Wczytuje struktury z pliku XYZ, zakładając energię w polu info."""
    print(f"Wczytywanie struktur z pliku: {file_path}")
    structures = ase.io.read(file_path, index=':')
    # Konwersja na obiekty Data z torch_geometric dla DataLoader
    data_list = []
    for atoms in structures:
        data = Data(
            atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            pos=torch.tensor(atoms.get_positions(), dtype=torch.float),
            energy_ccsdt=torch.tensor([atoms.info.get('energy')], dtype=torch.float)
        )
        data_list.append(data)
    print(f"Pomyślnie wczytano {len(data_list)} struktur.")
    return data_list

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Wykonuje jedną epokę treningu."""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Trening"):
        batch = batch.to(device)
        
        predictions = model(batch.to_dict())
        predicted_base_energy = predictions["base_energy_dft"]
        predicted_delta = predictions["delta_energy"]

        target_delta = batch.energy_ccsdt.view_as(predicted_base_energy) - predicted_base_energy

        loss = loss_fn(predicted_delta, target_delta.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__ == "__main__":
    MODEL_PATH = 'MACE_model_swa.model'
    DATASET_FILE = 'dummy_dataset.xyz' 
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2
    EPOCHS = 10
    
    if not os.path.exists(DATASET_FILE):
        print(f"Nie znaleziono pliku '{DATASET_FILE}'. Tworzenie przykładowego zbioru...")
        dummy_xyz_content = """2
energy=-1.0
O 0.0 0.0 0.0
H 0.0 0.0 1.0
---
3
energy=-2.5
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
"""
        with open(DATASET_FILE, 'w') as f:
            f.write(dummy_xyz_content)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    base_model = torch.load(MODEL_PATH, map_location=device, weights_only=False).float()
    dual_model = DualReadoutMACE(base_model).to(device)
    
    dataset = prepare_dataset_from_xyz(DATASET_FILE)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(dual_model.delta_readout.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("\nRozpoczęcie treningu nowej głowy 'delta_readout'...")
    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(dual_model, train_dataloader, optimizer, loss_fn, device)
        print(f"Epoka [{epoch+1}/{EPOCHS}], Średnia strata treningowa: {avg_train_loss:.6f}")

    output_path = 'delta_readout_trained.pt'
    torch.save(dual_model.delta_readout.state_dict(), output_path)
    print(f"\nTrening zakończony. Wagi nowej głowy zapisano w pliku: '{output_path}'")
