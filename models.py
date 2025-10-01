import torch
import torch.nn as nn
from typing import Dict

class DualReadoutMACE(nn.Module):
    """
    Klasa opakowująca istniejący, załadowany model MACE z dodaną drugą,
    trenowalną głową (readout) do nauki poprawki Δ.
    """
    def __init__(self, base_mace_model: nn.Module):
        super().__init__()
        self.features = None
        self.mace_model = base_mace_model

        print("Zamrażanie parametrów całego modelu bazowego MACE...")
        for param in self.mace_model.parameters():
            param.requires_grad = False
        self.mace_model.eval()

        self.base_readout = self.mace_model.readouts
        
        num_features = self.base_readout[0].linear.irreps_in.dim
        print(f"Wykryto {num_features} cech wejściowych do głowy (readout).")

        self.delta_readout = nn.Linear(num_features, 1, bias=False)
        self.base_readout.register_forward_hook(self._hook_fn)
        
        print("Nowa głowa 'delta_readout' została dodana.")

    def _hook_fn(self, module, input, output):
        self.features = input[0]

    def forward(self, data: Dict[str, torch.Tensor], compute_force: bool = False) -> Dict[str, torch.Tensor]:
        if compute_force:
            data["positions"].requires_grad_(True)
            
        _ = self.mace_model(data)

        if self.features is None:
            raise RuntimeError("Hak nie przechwycił cech atomowych.")
        
        atomic_features = self.features
        self.features = None

        base_atomic_energies = self.base_readout(atomic_features)
        base_energy = torch.sum(base_atomic_energies, dim=-2)

        delta_atomic_energies = self.delta_readout(atomic_features)
        delta_energy = torch.sum(delta_atomic_energies, dim=-2)

        final_energy = base_energy + delta_energy
        
        output_data = {}

        if compute_force:
            forces = -torch.autograd.grad(
                outputs=final_energy.sum(),
                inputs=data["positions"],
                create_graph=False, 
                retain_graph=False,
            )[0]
            output_data["forces"] = forces.detach()
        
        output_data["energy"] = final_energy.detach()

        return output_data
