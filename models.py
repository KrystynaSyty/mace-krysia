import torch
import torch.nn as nn
from typing import Dict

class DualReadoutMACE(nn.Module):
    """
    Finalna, poprawiona wersja klasy modelu.
    """
    def __init__(self, base_mace_model: nn.Module):
        super().__init__()
        self.features = None
        self.mace_model = base_mace_model

        print("Zamrażanie parametrów całego modelu bazowego MACE...")
        for param in self.mace_model.parameters():
            param.requires_grad = False
        # self.mace_model.eval() # Removed: Let the training script handle train/eval mode

        if not hasattr(self.mace_model, 'products') or not isinstance(self.mace_model.products, nn.ModuleList) or len(self.mace_model.products) == 0:
            raise AttributeError("Model MACE nie ma atrybutu 'products'. Hak nie może zostać zarejestrowany.")

        num_features = self.mace_model.readouts[0].linear.irreps_in.dim
        print(f"Wykryto {num_features} cech wejściowych do głowy (readout).")

        self.delta_readout = nn.Linear(num_features, 1, bias=False)
        
        torch.nn.init.zeros_(self.delta_readout.weight)
        print("Wagi nowej głowy 'delta_readout' zostały zainicjalizowane zerami.")

        feature_extractor_layer = self.mace_model.products[-1]
        feature_extractor_layer.register_forward_hook(self._hook_fn)
        print("Hak poprawnie zarejestrowany na ostatnim bloku 'product'.")

    def to(self, *args, **kwargs):
        """Upewnia się, że nowa warstwa jest na tym samym urządzeniu i ma ten sam typ co reszta modelu."""
        super().to(*args, **kwargs)
        # Dopasuj typ i urządzenie nowej warstwy do modelu bazowego
        dtype = next(self.mace_model.parameters()).dtype
        device = next(self.mace_model.parameters()).device
        self.delta_readout.to(device=device, dtype=dtype)
        return self

    def _hook_fn(self, module, input_data, output_data):
        self.features = output_data

    def forward(self, data: Dict[str, torch.Tensor], compute_force: bool = False) -> Dict[str, torch.Tensor]:
        if compute_force:
            data["positions"].requires_grad_(True)

        # A forward pass is needed to trigger the hook and get features
        base_output = self.mace_model(data, compute_force=False)
        # The base_energy doesn't need to be detached. Gradients won't flow
        # to the base model anyway since its parameters have requires_grad=False.
        base_energy = base_output["energy"]

        if self.features is None:
            raise RuntimeError("Hak nie przechwycił cech atomowych. Upewnij się, że zrestartowałeś/aś kernel.")

        delta_atomic_energies = self.delta_readout(self.features)
        delta_energy = torch.sum(delta_atomic_energies, dim=-2)
        
        self.features = None
        
        final_energy = base_energy + delta_energy

        output_data = {}
        if compute_force:
            forces = -torch.autograd.grad(
                outputs=final_energy.sum(),
                inputs=data["positions"],
                create_graph=self.training, # Create graph if in training mode
                retain_graph=self.training,
            )[0]
            output_data["forces"] = forces

        # Return the final energy attached to the graph for training
        output_data["energy"] = final_energy
        return output_data

