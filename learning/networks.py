"""
Neural network architecture utilities for deep reinforcement learning.
"""

import numpy as np
import torch
from torch import nn
from typing import Union, List

Activation = Union[str, nn.Module]

_STR_TO_ACTIVATION = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = 'tanh',
    output_activation: Activation = 'identity',
) -> nn.Module:
    """Builds a standard feedforward multi-layer perceptron (MLP)."""
    if isinstance(activation, str):
        activation = _STR_TO_ACTIVATION[activation]
    if isinstance(output_activation, str):
        output_activation = _STR_TO_ACTIVATION[output_activation]

    layers: List[nn.Module] = []
    in_size = input_size
    
    # Construct hidden layers
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
        
    # Construct output layer
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    
    return nn.Sequential(*layers)

# --- Hardware & Tensor Management ---

def get_device() -> torch.device:
    """Returns the optimal available PyTorch hardware device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely detaches a tensor and moves it to a numpy array."""
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a numpy array to a float tensor on the target device."""
    return torch.from_numpy(array.astype(np.float32)).to(device)