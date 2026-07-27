"""
Behavioral Cloning Trainer.
Orchestrates the loading of expert demonstrations and the supervised 
training loop for continuous-control MuJoCo environments.
"""

import pickle
import time
import numpy as np
import gymnasium as gym

from learning.policy import ImitationPolicy
from learning.buffer import ReplayBuffer
from learning import utils

class BehavioralCloningTrainer:
    """
    Executes the training and evaluation pipeline for Imitation Learning.
    """
    def __init__(
        self,
        env_name: str,
        policy: ImitationPolicy,
        batch_size: int = 1000,
        eval_batch_size: int = 5000,
        max_ep_len: int = 1000,
        seed: int = 42
    ):
        self.env_name = env_name
        self.policy = policy
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_ep_len = max_ep_len
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        
        # Initialize the Gymnasium environment
        # Ant tasks require contact forces to match expert observation space (111 dims)
        if 'Ant' in env_name:
            self.env = gym.make(env_name, use_contact_forces=True, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name, render_mode="rgb_array")
            
        self.buffer = ReplayBuffer()

    def load_expert_data(self, expert_data_path: str):
        """Bootstraps the replay buffer with expert demonstrations."""
        print(f"Loading expert demonstrations from {expert_data_path}...")
        with open(expert_data_path, 'rb') as f:
            expert_paths = pickle.load(f)
            
        self.buffer.add_rollouts(expert_paths)
        total_steps = sum([len(path['reward']) for path in expert_paths])
        print(f"Successfully loaded {len(expert_paths)} trajectories ({total_steps} total transitions).")

    def train(self, num_training_steps: int):
        """Executes the supervised learning loop over the expert dataset."""
        print("\nBeginning Behavioral Cloning Training...")
        start_time = time.time()
        
        training_losses = []
        
        for step in range(num_training_steps):
            # 1. Sample a mini-batch of expert transitions
            ob_batch, ac_batch, _, _, _ = self.buffer.sample_random_data(self.batch_size)
            
            # 2. Execute a gradient descent step
            loss = self.policy.update(ob_batch, ac_batch)
            training_losses.append(loss)
            
            # 3. Log progress
            if step % 500 == 0 or step == num_training_steps - 1:
                avg_loss = np.mean(training_losses[-100:])
                elapsed = time.time() - start_time
                print(f"Step {step:05d} | Avg Loss (last 100): {avg_loss:.4f} | Time Elapsed: {elapsed:.1f}s")

    def evaluate(self, render: bool = False):
        """Rolls out the trained policy in the environment to calculate performance metrics."""
        print(f"\nEvaluating policy over {self.eval_batch_size} timesteps...")
        
        eval_paths, total_steps = utils.sample_trajectories(
            self.env, 
            self.policy, 
            self.eval_batch_size, 
            self.max_ep_len, 
            render=render
        )
        
        returns = [path["reward"].sum() for path in eval_paths]
        ep_lengths = [len(path["reward"]) for path in eval_paths]
        
        print("-" * 40)
        print("Evaluation Metrics:")
        print(f"Average Return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
        print(f"Max Return:     {np.max(returns):.2f}")
        print(f"Min Return:     {np.min(returns):.2f}")
        print(f"Average Ep Len: {np.mean(ep_lengths):.1f}")
        print("-" * 40)