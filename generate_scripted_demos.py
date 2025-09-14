#!/usr/bin/env python3
"""
Generate scripted demonstration trajectories for mobile robot environment.
This creates basic demo trajectories using a simple scripted policy.
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

def simple_grasping_policy(obs, step_count, max_steps=200):
    """
    A simple scripted policy for banana grasping.
    This is just an example - you should replace with a better policy.
    """
    # Get proprioceptive state
    if 'state' in obs:
        joint_positions = obs['state'][:7]  # First 7 are joint positions
    else:
        joint_positions = np.zeros(7)
    
    # Simple scripted behavior
    action = np.zeros(7)
    
    # Phase 1: Move to approach position (steps 0-50)
    if step_count < 50:
        # Move arm to a reasonable position above banana
        target_joints = np.array([0.0, -0.5, 0.5, -1.0, 0.0, 0.5, 0.0])
        action = (target_joints - joint_positions) * 0.1
        
    # Phase 2: Lower arm (steps 50-100)
    elif step_count < 100:
        action[1] = 0.05  # Lower shoulder
        action[2] = -0.05  # Extend elbow
        
    # Phase 3: Approach banana (steps 100-150)
    elif step_count < 150:
        action[0] = 0.02  # Small base movement
        action[3] = 0.03  # Wrist adjustment
        
    # Phase 4: Close gripper (steps 150-180)
    elif step_count < 180:
        action[6] = -0.05  # Close gripper
        
    # Phase 5: Lift (steps 180-200)
    else:
        action[1] = -0.03  # Lift shoulder
        action[6] = -0.05  # Keep gripper closed
    
    # Add some randomness for variation
    noise = np.random.normal(0, 0.01, action.shape)
    action += noise
    
    # Clip to action space bounds
    action = np.clip(action, -1.0, 1.0)
    
    return action

def generate_scripted_demo(env, trajectory_id: int, max_steps: int = 200) -> Dict[str, Any]:
    """
    Generate a single scripted demonstration trajectory.
    """
    print(f"🤖 Generating scripted trajectory {trajectory_id + 1}")
    
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'next_observations': []
    }
    
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < max_steps:
        # Store current observation
        trajectory['observations'].append(obs)
        
        # Get action from scripted policy
        action = simple_grasping_policy(obs, step_count, max_steps)
        
        # Take environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['terminals'].append(done)
        trajectory['next_observations'].append(next_obs)
        
        total_reward += reward
        step_count += 1
        obs = next_obs
    
    print(f"✅ Generated trajectory {trajectory_id + 1}: {step_count} steps, reward: {total_reward:.2f}")
    
    # Convert lists to numpy arrays
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory

def generate_scripted_demonstrations(env_name: str = 'MobileRobotEnv-v0', 
                                   num_demos: int = 20, 
                                   save_path: str = 'mobile_robot_scripted_demos.pkl',
                                   max_steps_per_demo: int = 200):
    """
    Generate multiple scripted demonstration trajectories.
    """
    print(f"🤖 Generating {num_demos} scripted demonstrations for {env_name}")
    
    # Create environment
    env = gym.make(env_name)
    
    # Apply transformations
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
            trajectory = generate_scripted_demo(env, i, max_steps_per_demo)
            trajectories.append(trajectory)
            
        except Exception as e:
            print(f"❌ Error generating trajectory {i + 1}: {e}")
            continue
    
    env.close()
    
    if trajectories:
        # Save trajectories
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(trajectories, f)
        
        print(f"\n💾 Generated and saved {len(trajectories)} scripted demonstrations!")
        print(f"📁 Saved to: {save_path}")
        
        # Print statistics
        total_steps = sum(len(traj['actions']) for traj in trajectories)
        avg_steps = total_steps / len(trajectories)
        avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories])
        
        print(f"\n📊 Statistics:")
        print(f"   Trajectories: {len(trajectories)}")
        print(f"   Total steps: {total_steps}")
        print(f"   Average steps per trajectory: {avg_steps:.1f}")
        print(f"   Average total reward: {avg_reward:.2f}")
        
        # Check trajectory format
        sample_traj = trajectories[0]
        print(f"   Observation keys: {list(sample_traj['observations'][0].keys())}")
        print(f"   Action shape: {sample_traj['actions'][0].shape}")
        
        print(f"\n🚀 Ready for training! Use:")
        print(f"   bash run_learner.sh --demo_path {save_path}")
        
        return save_path
    else:
        print("❌ No trajectories generated.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate scripted demonstrations for mobile robot")
    parser.add_argument("--env", default="MobileRobotEnv-v0", help="Environment name")
    parser.add_argument("--num_demos", type=int, default=20, help="Number of demo trajectories to generate")
    parser.add_argument("--save_path", default="mobile_robot_scripted_demos.pkl", help="Path to save demo trajectories")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per demonstration")
    
    args = parser.parse_args()
    
    demo_path = generate_scripted_demonstrations(
        env_name=args.env,
        num_demos=args.num_demos,
        save_path=args.save_path,
        max_steps_per_demo=args.max_steps
    )
