"""
Quickstart Demo: Behavioral Cloning (Imitation Learning)

This script trains a Multi-Layer Perceptron (MLP) policy to mimic 
expert locomotion behavior on the high-dimensional MuJoCo Ant-v4 environment.
"""

import os
import torch
from learning.policy import ImitationPolicy
from learning.trainer import BehavioralCloningTrainer

def run_demo():
    print("="*50)
    print("Initializing Imitation Learning Pipeline")
    print("="*50)
    
    # Environment parameters for MuJoCo Ant-v4
    env_name = "Ant-v4"
    ob_dim = 111  # Observation space dimension (with contact forces)
    ac_dim = 8    # Action space dimension (8 leg joints)
    
    # Initialize the neural network policy
    print(f"Building MLP Policy (Inputs: {ob_dim}, Outputs: {ac_dim})...")
    policy = ImitationPolicy(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        n_layers=2,
        size=64,
        discrete=False,
        learning_rate=1e-3
    )
    
    # Initialize the Trainer
    trainer = BehavioralCloningTrainer(
        env_name=env_name,
        policy=policy,
        batch_size=1000,
        eval_batch_size=3000, 
        max_ep_len=1000
    )
    
    # NOTE: You must provide a valid expert data .pkl file to train.
    # Replace this path with the actual location of your expert data.
    expert_data_path = "data/expert_data_ant.pkl" 
    
    if not os.path.exists(expert_data_path):
        print(f"\n[ERROR] Expert data not found at {expert_data_path}.")
        print("Please ensure your expert .pkl file is placed in the correct directory to run the training loop.")
        return

    # Execute the Behavioral Cloning Pipeline
    trainer.load_expert_data(expert_data_path)
    trainer.train(num_training_steps=2000)
    
    # Evaluate the learned policy
    trainer.evaluate(render=False)
    
    # Save the final model weights
    os.makedirs("models", exist_ok=True)
    model_path = "models/bc_ant_policy.pt"
    policy.save(model_path)
    print(f"\nSaved trained policy weights to: {model_path}")

if __name__ == "__main__":
    run_demo()