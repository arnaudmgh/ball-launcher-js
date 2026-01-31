import numpy as np
from ball_launcher_env import BallLauncherEnv
import json
import os

def train_cem(num_iterations=50, population_size=100, elite_frac=0.1):
    env = BallLauncherEnv(num_targets=3)
    
    # Initialize distribution: mean and std for [velocity, angle]
    # Starting with a broad search
    mean = np.array([800.0, np.pi/4])
    std = np.array([400.0, 0.5])
    
    # Fix the target configuration for training this "level"
    # To make it truly "get good", we could train on random configs, 
    # but CEM usually finds a solution for a specific setup.
    obs, info = env.reset()
    target_configs = []
    for t in env.targets:
        target_configs.append({'x': t['x'], 'y': t['y'], 'r': t['r']})
    
    print(f"Target Configuration: {target_configs}")
    
    best_overall_reward = -float('inf')
    best_overall_action = None
    
    for i in range(num_iterations):
        # Sample population
        # Use truncated normal or just clipping to keep within bounds
        samples = np.random.normal(mean, std, size=(population_size, 2))
        
        rewards = []
        for action in samples:
            # Clip action to environment bounds
            clipped_action = np.clip(
                action, 
                env.action_space.low, 
                env.action_space.high
            )
            
            # Reset to same target config for evaluation
            env.reset(target_configs=target_configs)
            _, reward, _, _, _ = env.step(clipped_action)
            rewards.append(reward)
            
            if reward > best_overall_reward:
                best_overall_reward = reward
                best_overall_action = clipped_action.tolist()
        
        rewards = np.array(rewards)
        
        # Select elites
        n_elites = int(population_size * elite_frac)
        elite_indices = rewards.argsort()[-n_elites:]
        elites = samples[elite_indices]
        
        # Update distribution
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-6 # Add epsilon to avoid zero std
        
        print(f"Iteration {i+1}/{num_iterations} | Max Reward: {rewards.max():.2f} | Mean: {mean}")
        
        if rewards.max() >= 500: # Threshold for "good enough" (all hit + shaped)
             print("Reached high reward threshold!")
             # We could break early, but let's keep refining
             
    # Save best solution and config
    result = {
        'target_configs': target_configs,
        'best_action': best_overall_action,
        'best_reward': best_overall_reward
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/best_solution.json', 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"\nTraining Complete. Best Reward: {best_overall_reward:.2f}")
    print(f"Best Action: Velocity={best_overall_action[0]:.2f}, Angle={best_overall_action[1]:.2f} rad")

if __name__ == "__main__":
    train_cem()
