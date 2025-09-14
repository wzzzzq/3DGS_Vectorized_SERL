#!/usr/bin/env python3
"""
Script to record demonstration trajectories for mobile robot environment.
This creates demo trajectories that can be used for training with SERL.
"""

import gym
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from gymnasium.wrappers import TransformObservation
import gymnasium.spaces as spaces

def transform_obs(obs):
    """Transform observation to match training format"""
    new_obs = obs.copy()
    if 'rgb' in new_obs:
        new_obs['front'] = new_obs.pop('rgb')
    return new_obs

def record_demo_trajectory(env, trajectory_id: int) -> Dict[str, Any]:
    """
    Record a single demonstration trajectory.
    Returns trajectory in the format expected by SERL.
    """
    print(f"\n🎮 Recording trajectory {trajectory_id + 1}")
    print("Controls:")
    print("  - Use your input method to control the robot")
    print("  - Press 'q' to quit current trajectory")
    print("  - Press 'r' to reset and restart trajectory")
    print("  - The robot should grasp the banana successfully")
    
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'next_observations': []
    }
    
    obs, info = env.reset()
    obs = transform_obs(obs)
    done = False
    step_count = 0
    total_reward = 0
    
    print(f"🚀 Starting trajectory {trajectory_id + 1}. Control the robot to grasp the banana!")
    
    while not done:
        # Store current observation
        trajectory['observations'].append(obs)
        
        # For demo recording, you could:
        # 1. Use keyboard/gamepad input
        # 2. Use pre-recorded actions
        # 3. Use a scripted policy
        # 4. Manual control through GUI
        
        # Example: Random action (replace with your control method)
        action = env.action_space.sample()
        
        # For manual control, you might want to pause here and get input
        print(f"Step {step_count}: Taking action {action}")
        
        # Take environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = transform_obs(next_obs)
        done = terminated or truncated
        
        # Store transition
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['terminals'].append(done)
        trajectory['next_observations'].append(next_obs)
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        # Optional: Add some basic success detection
        if step_count >= 1000:  # Max episode length
            print("⏰ Maximum episode length reached")
            break
            
        # You can add manual control logic here
        # For example, reading keyboard input or waiting for user input
    
    print(f"✅ Trajectory {trajectory_id + 1} completed!")
    print(f"   Steps: {step_count}, Total reward: {total_reward:.2f}")
    
    # Convert lists to numpy arrays
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory

def record_demonstrations(env_name: str = 'MobileRobotEnv-v0', 
                         num_demos: int = 20, 
                         save_path: str = 'mobile_robot_demos.pkl'):
    """
    Record multiple demonstration trajectories and save them.
    """
    print(f"🎯 Recording {num_demos} demonstration trajectories for {env_name}")
    
    # Create environment
    env = gym.make(env_name)
    
    # Apply same transformations as training
    env = TransformObservation(env, transform_obs)
    
    # Create transformed observation space
    original_obs_space = env.observation_space
    if isinstance(original_obs_space, spaces.Dict) and 'rgb' in original_obs_space.spaces:
        transformed_spaces = original_obs_space.spaces.copy()
        transformed_spaces['front'] = transformed_spaces.pop('rgb')
        transformed_obs_space = spaces.Dict(transformed_spaces)
        env.observation_space = transformed_obs_space
    
    trajectories = []
    
    for i in range(num_demos):
        try:
            trajectory = record_demo_trajectory(env, i)
            trajectories.append(trajectory)
            
            # Ask user if they want to continue
            if i < num_demos - 1:
                user_input = input(f"\n📝 Recorded {i + 1}/{num_demos} trajectories. Continue? (y/n/q): ").lower()
                if user_input in ['n', 'q', 'quit', 'no']:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⏹️  Recording interrupted. Saved {len(trajectories)} trajectories so far.")
            break
        except Exception as e:
            print(f"❌ Error recording trajectory {i + 1}: {e}")
            continue
    
    env.close()
    
    if trajectories:
        # Save trajectories
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(trajectories, f)
        
        print(f"\n💾 Saved {len(trajectories)} demonstration trajectories to {save_path}")
        print(f"📊 Trajectory stats:")
        
        # Print statistics
        total_steps = sum(len(traj['actions']) for traj in trajectories)
        avg_steps = total_steps / len(trajectories)
        avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories])
        
        print(f"   Total trajectories: {len(trajectories)}")
        print(f"   Total steps: {total_steps}")
        print(f"   Average steps per trajectory: {avg_steps:.1f}")
        print(f"   Average total reward: {avg_reward:.2f}")
        
        # Check observation and action shapes
        sample_traj = trajectories[0]
        print(f"   Observation keys: {list(sample_traj['observations'][0].keys())}")
        print(f"   Action shape: {sample_traj['actions'][0].shape}")
        
        return save_path
    else:
        print("❌ No trajectories recorded.")
        return None

def load_and_inspect_demos(demo_path: str):
    """Load and inspect saved demonstration trajectories."""
    if not os.path.exists(demo_path):
        print(f"❌ Demo file {demo_path} not found.")
        return
    
    with open(demo_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"📋 Inspecting {demo_path}")
    print(f"   Number of trajectories: {len(trajectories)}")
    
    if trajectories:
        sample_traj = trajectories[0]
        print(f"   Sample trajectory length: {len(sample_traj['actions'])}")
        print(f"   Observation keys: {list(sample_traj['observations'][0].keys())}")
        print(f"   Action shape: {sample_traj['actions'][0].shape}")
        print(f"   Reward range: [{np.min([np.sum(t['rewards']) for t in trajectories]):.2f}, "
              f"{np.max([np.sum(t['rewards']) for t in trajectories]):.2f}]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Record demonstration trajectories for mobile robot")
    parser.add_argument("--env", default="MobileRobotEnv-v0", help="Environment name")
    parser.add_argument("--num_demos", type=int, default=20, help="Number of demo trajectories to record")
    parser.add_argument("--save_path", default="mobile_robot_demos.pkl", help="Path to save demo trajectories")
    parser.add_argument("--inspect", help="Path to demo file to inspect")
    
    args = parser.parse_args()
    
    if args.inspect:
        load_and_inspect_demos(args.inspect)
    else:
        demo_path = record_demonstrations(
            env_name=args.env,
            num_demos=args.num_demos,
            save_path=args.save_path
        )
        
        if demo_path:
            print(f"\n🎉 Demo recording complete! Use this file for training:")
            print(f"   bash run_learner.sh --demo_path {demo_path}")
