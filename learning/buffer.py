"""
Experience Replay Buffer for Imitation Learning.
Stores and samples expert trajectories for supervised learning.
"""

import numpy as np
from typing import List, Tuple, Dict

class ReplayBuffer:
    """
    Stores environment rollouts and provides aligned random sampling 
    for neural network mini-batch updates.
    """
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.paths: List[Dict[str, np.ndarray]] = []

        # Concatenated component arrays
        self.obs: np.ndarray = None
        self.acs: np.ndarray = None
        self.rews: np.ndarray = None
        self.next_obs: np.ndarray = None
        self.terminals: np.ndarray = None

    def __len__(self) -> int:
        return self.obs.shape[0] if self.obs is not None else 0

    def add_rollouts(self, paths: List[Dict[str, np.ndarray]]):
        """Adds a list of rollout dictionaries to the replay buffer."""
        self.paths.extend(paths)

        # Extract and concatenate new rollouts
        new_obs = np.concatenate([path["observation"] for path in paths])
        new_acs = np.concatenate([path["action"] for path in paths])
        new_rews = np.concatenate([path["reward"] for path in paths])
        new_next_obs = np.concatenate([path["next_observation"] for path in paths])
        new_terminals = np.concatenate([path["terminal"] for path in paths])

        if self.obs is None:
            self.obs = new_obs[-self.max_size:]
            self.acs = new_acs[-self.max_size:]
            self.rews = new_rews[-self.max_size:]
            self.next_obs = new_next_obs[-self.max_size:]
            self.terminals = new_terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, new_obs])[-self.max_size:]
            self.acs = np.concatenate([self.acs, new_acs])[-self.max_size:]
            self.rews = np.concatenate([self.rews, new_rews])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, new_next_obs])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, new_terminals])[-self.max_size:]

    def sample_random_data(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a randomized, aligned mini-batch of transitions."""
        assert (
            self.obs.shape[0] == self.acs.shape[0] == self.rews.shape[0] == 
            self.next_obs.shape[0] == self.terminals.shape[0]
        ), "Buffer arrays are out of alignment!"

        buffer_size = self.obs.shape[0]
        indices = np.random.permutation(buffer_size)[:batch_size]

        return (
            self.obs[indices],
            self.acs[indices],
            self.rews[indices],
            self.next_obs[indices],
            self.terminals[indices]
        )

    def sample_recent_data(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves the most recently added transitions."""
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:]
        )