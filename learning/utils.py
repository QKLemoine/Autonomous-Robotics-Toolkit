"""
Environment interaction utilities for executing policy rollouts.
"""

import time
import numpy as np
from typing import List, Dict, Any

def get_pathlength(path: Dict[str, np.ndarray]) -> int:
    """Returns the total number of steps in a rollout."""
    return len(path["reward"])

def sample_trajectory(
    env: Any, 
    policy: Any, 
    max_path_length: int, 
    render: bool = False, 
    render_mode: tuple = ('rgb_array')
) -> Dict[str, np.ndarray]:
    """
    Executes a single rollout of the policy in the environment.
    Returns a dictionary of the transition data.
    """
    ob, _ = env.reset()
    
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render())
            if 'human' in render_mode:
                env.render()
                if hasattr(env, 'model'):
                    time.sleep(env.model.opt.timestep)

        obs.append(ob)
        
        # Policy returns a batched action, extract the first one
        ac = policy.get_action(ob)[0]
        acs.append(ac)

        # Execute action in the environment (Modern Gymnasium API)
        ob, rew, terminated, truncated, _ = env.step(ac)
        
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # Check for episode termination or truncation
        rollout_done = terminated or truncated or steps >= max_path_length
        terminals.append(rollout_done)

        if rollout_done:
            break

    # Compile the trajectory into a single dictionary
    trajectory = {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32)
    }
    
    if image_obs:
        trajectory["image_obs"] = np.stack(image_obs, axis=0).astype(np.uint8)
        
    return trajectory

def sample_trajectories(
    env: Any, 
    policy: Any, 
    min_timesteps_per_batch: int, 
    max_path_length: int, 
    render: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """
    Collects rollouts until a minimum number of environment steps is reached.
    """
    timesteps_this_batch = 0
    paths = []
    
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
        
    return paths, timesteps_this_batch

def sample_n_trajectories(
    env: Any, 
    policy: Any, 
    ntraj: int, 
    max_path_length: int, 
    render: bool = False
) -> List[Dict[str, np.ndarray]]:
    """Collects exactly `ntraj` rollouts."""
    return [sample_trajectory(env, policy, max_path_length, render) for _ in range(ntraj)]