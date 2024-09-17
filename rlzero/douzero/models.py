"""This file defines PyTorch models for predicting actions in a game-like
environment.

We include three models: one for the "landlord" and two for the "farmers". All
three models are wrapped into a convenient class that provides methods for
evaluation, memory sharing, and parameter access.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn


class BaseModel(nn.Module):
    """Base model for DouDiZhu game.

    This model uses an LSTM to process sequential data and fully connected
    layers for further processing and prediction.
    """

    def __init__(
        self,
        input_dim: int = 162,
        hidden_dim: int = 128,
        special_dim: int = 373,
        output_dim: int = 512,
    ) -> None:
        """Initializes the BaseModel with LSTM and dense layers.

        Args:
            input_dim (int): Input dimension for the LSTM.
            hidden_dim (int): Hidden dimension for the LSTM.
            special_dim (int): Special dimension for the dense layers.
            output_dim (int): Output dimension for the dense layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim + special_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        return_value: bool = False,
        flags: Optional[object] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """Forward pass for the BaseModel.

        Args:
            z (torch.Tensor): Sequential input data for the LSTM.
            x (torch.Tensor): Additional input data concatenated after LSTM output.
            return_value (bool): If True, returns values; otherwise, returns predicted action.
            flags (Optional[object]): Optional flags for exploration strategies.

        Returns:
            Dict[str, Union[torch.Tensor, int]]: Output from the respective model's forward method.
        """
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.fc_layers(x)

        if return_value:
            return dict(values=x)
        else:
            action = self._select_action(x, flags)
            return dict(action=action)

    def _select_action(self, x: torch.Tensor, flags: Optional[object]) -> int:
        if (flags is not None and flags.exp_epsilon > 0
                and np.random.rand() < flags.exp_epsilon):
            return torch.randint(x.shape[0], (1, ))[0]
        else:
            return torch.argmax(x, dim=0)[0]


class LandlordLstmModel(BaseModel):
    """LSTM-based model for predicting actions or values in a landlord-like
    scenario."""

    def __init__(self) -> None:
        """Initializes the LandlordLstmModel with LSTM and dense layers."""
        super().__init__(input_dim=162, special_dim=373, output_dim=512)


class FarmerLstmModel(BaseModel):
    """LSTM-based model for predicting actions or values in a farmer-like
    scenario."""

    def __init__(self) -> None:
        super().__init__(input_dim=162, special_dim=484, output_dim=512)


# Dictionary to map different model roles to their respective classes
model_dict = {
    'landlord': LandlordLstmModel,
    'landlord_up': FarmerLstmModel,
    'landlord_down': FarmerLstmModel,
}


class DouDiZhuModel(nn.Module):
    """Wrapper for handling multiple models (landlord, farmer) and providing
    interfaces for shared memory and evaluation."""

    def __init__(self, device: Union[int, str] = 0):
        """Initializes the Model class with models for each role and places
        them on the specified device.

        Args:
            device (Union[int, str]): Device for model computation, either a GPU index or 'cpu'.
        """
        self.models = {}
        device_str = 'cpu' if device == 'cpu' else f'cuda:{device}'
        device = torch.device(device_str)

        # Initialize models for different roles and move to the specified device
        self.models['landlord'] = LandlordLstmModel(input_dim=162,
                                                    special_dim=373,
                                                    output_dim=512).to(device)
        self.models['landlord_up'] = FarmerLstmModel(input_dim=162,
                                                     special_dim=484,
                                                     output_dim=512).to(device)
        self.models['landlord_down'] = FarmerLstmModel(
            input_dim=162, special_dim=484, output_dim=512).to(device)

    def forward(
        self,
        position: str,
        z: torch.Tensor,
        x: torch.Tensor,
        training: bool = False,
        flags: Optional[object] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """Forward pass for the selected model based on the position (landlord
        or farmer).

        Args:
            position (str): Role of the model ('landlord', 'landlord_up', 'landlord_down').
            z (torch.Tensor): Sequential input data for the LSTM.
            x (torch.Tensor): Additional input data concatenated after LSTM output.
            training (bool): Whether the model is in training mode.
            flags (Optional[object]): Optional flags for exploration strategies.

        Returns:
            Dict[str, Union[torch.Tensor, int]]: Output from the respective model's forward method.
        """
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self) -> None:
        """Shares memory for all models, allowing multi-processing in
        PyTorch."""
        for model in self.models.values():
            model.share_memory()

    def eval(self) -> None:
        """Sets all models to evaluation mode."""
        for model in self.models.values():
            model.eval()

    def parameters(self, position: str):
        """Returns the parameters of the model corresponding to the given
        position."""
        return self.models[position].parameters()

    def get_model(self, position: str) -> nn.Module:
        """Returns the model corresponding to the given position."""
        return self.models[position]

    def get_models(self) -> Dict[str, nn.Module]:
        """Returns a dictionary of all models."""
        return self.models

    def save(self, path: str) -> None:
        torch.save(
            {
                'landlord': self.models['landlord'].state_dict(),
                'landlord_up': self.models['landlord_up'].state_dict(),
                'landlord_down': self.models['landlord_down'].state_dict(),
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = torch.load(path)
        self.models['landlord'].load_state_dict(checkpoint['landlord'])
        self.models['landlord_up'].load_state_dict(checkpoint['landlord_up'])
        self.models['landlord_down'].load_state_dict(
            checkpoint['landlord_down'])
