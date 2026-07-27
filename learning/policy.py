"""
Behavioral Cloning Policy.
Uses supervised learning to mimic expert continuous-control trajectories.
"""

import numpy as np
import torch
from torch import nn

from learning.networks import build_mlp, get_device, from_numpy, to_numpy

class ImitationPolicy(nn.Module):
    """
    A neural network policy that maps state observations to actions.
    Optimized via Supervised Learning against an expert dataset.
    """
    def __init__(
        self, 
        ac_dim: int, 
        ob_dim: int,
        n_layers: int, 
        size: int, 
        discrete: bool = False, 
        learning_rate: float = 1e-4
    ):
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.discrete = discrete
        self.device = get_device()

        # Build the architecture based on the action space
        if self.discrete:
            self.logits_net = build_mlp(ob_dim, ac_dim, n_layers, size).to(self.device)
            self.mean_net = None
            self.logstd_param = None
        else:
            self.logits_net = None
            self.mean_net = build_mlp(ob_dim, ac_dim, n_layers, size).to(self.device)
            self.logstd_param = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=self.device)
            )

        # Encapsulate optimization directly inside the policy
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.continuous_loss = nn.MSELoss()
        self.discrete_loss = nn.CrossEntropyLoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Samples an action from the policy distribution (Used for RL/Exploration)."""
        if self.discrete:
            logits = self.logits_net(obs)
            return torch.distributions.Categorical(logits=logits).sample()
        else:
            mean = self.mean_net(obs)
            logstd = torch.exp(self.logstd_param)
            return torch.distributions.Normal(mean, logstd).sample()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Predicts a deterministic action for environment rollouts."""
        if len(obs.shape) < 2:
            obs = obs[None]  # Inject batch dimension if missing
        
        obs_tensor = from_numpy(obs, self.device)
        
        with torch.no_grad():
            if self.discrete:
                logits = self.logits_net(obs_tensor)
                action = torch.distributions.Categorical(logits=logits).sample()
            else:
                # For continuous evaluation, bypass exploration noise and use the mean
                action = self.mean_net(obs_tensor)
                
        return to_numpy(action)

    def update(self, observations: np.ndarray, actions: np.ndarray) -> float:
        """Executes a single supervised learning backpropagation step."""
        obs_tensor = from_numpy(observations, self.device)
        act_tensor = from_numpy(actions, self.device)

        if self.discrete:
            predicted_logits = self.logits_net(obs_tensor)
            loss = self.discrete_loss(predicted_logits, act_tensor.squeeze().long())
        else:
            # Train against the differentiable mean network output
            predicted_mean = self.mean_net(obs_tensor)
            loss = self.continuous_loss(predicted_mean, act_tensor)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, filepath: str):
        """Serializes the model state dictionary."""
        torch.save(self.state_dict(), filepath)