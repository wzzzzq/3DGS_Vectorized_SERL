#!/usr/bin/env python3
"""
Interactive demo recording script for mobile robot environment.
This allows you to manually control the robot and record successful demonstrations.
"""

import gym
import numpy as np
import pickle
import os
import cv2
from typing import List, Dict, Any, Optional
from gymnasium.wrappers import TransformObservation
import gymnasium.spaces as spaces
import time

class ManualController:
    """
    Manual controller for recording demonstrations.
    Uses keyboard input to control the robot.
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.current_action = np.zeros(action_space.shape)
        self.action_scale = 0.1  # Scale factor for manual control
        
        print("\n🎮 Manual Control Instructions:")
        print("  Joint 0 (Base): A/D (left/right)")
        print("  Joint 1 (Shoulder): W/S (up/down)")  
        print("  Joint 2 (Elbow): Q/E (up/down)")
        print("  Joint 3 (Wrist1): R/F (rotate)")
        print("  Joint 4 (Wrist2): T/G (rotate)")
        print("  Joint 5 (Wrist3): Y/H (rotate)")
        print("  Joint 6 (Gripper): U/I (open/close)")
        print("  SPACE: Reset action to zero")
        print("  ENTER: Execute current action")
        print("  ESC: End trajectory")
        print("  'r': Reset environment")
        print("  's': Save successful trajectory")
        print("  'q': Quit recording")
    
    def get_action_from_key(self, key):
        """Convert keyboard input to action."""
        action = np.zeros_like(self.current_action)
        
        # Joint 0 (Base rotation)
        if key == ord('a') or key == ord('A'):
            action[0] = -self.action_scale
        elif key == ord('d') or key == ord('D'):
            action[0] = self.action_scale
            
        # Joint 1 (Shoulder)
        elif key == ord('w') or key == ord('W'):
            action[1] = self.action_scale
        elif key == ord('s') or key == ord('S'):
            action[1] = -self.action_scale
            
        # Joint 2 (Elbow)
        elif key == ord('q') or key == ord('Q'):
            action[2] = self.action_scale
        elif key == ord('e') or key == ord('E'):
            action[2] = -self.action_scale
            
        # Joint 3 (Wrist1)
        elif key == ord('r') or key == ord('R'):
            action[3] = self.action_scale
        elif key == ord('f') or key == ord('F'):
            action[3] = -self.action_scale
            
        # Joint 4 (Wrist2)
        elif key == ord('t') or key == ord('T'):
            action[4] = self.action_scale
        elif key == ord('g') or key == ord('G'):
            action[4] = -self.action_scale
            
        # Joint 5 (Wrist3)
        elif key == ord('y') or key == ord('Y'):
            action[5] = self.action_scale
        elif key == ord('h') or key == ord('H'):
            action[5] = -self.action_scale
            
        # Joint 6 (Gripper)
        elif key == ord('u') or key == ord('U'):
            action[6] = self.action_scale
        elif key == ord('i') or key == ord('I'):
            action[6] = -self.action_scale
            
        # Reset action
        elif key == ord(' '):
            self.current_action = np.zeros_like(self.current_action)
            return None
            
        return action

def display_observation(obs, window_name="Robot Camera"):
    """Display the robot's camera observation."""
    if 'front' in obs:
        img = obs['front']
        if img.max() <= 1.0:  # Normalize if needed
            img = (img * 255).astype(np.uint8)
        
        # Add text overlay with control instructions
        img_display = img.copy()
        cv2.putText(img_display, "Press keys to control robot", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_display, "ESC: End trajectory, 's': Save, 'q': Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, img_display)
        return cv2.waitKey(1) & 0xFF
    return -1

def record_manual_demo(env, trajectory_id: int, controller: ManualController) -> Optional[Dict[str, Any]]:
    """
    Record a single demonstration trajectory with manual control.
    """
    print(f"\n🎮 Recording trajectory {trajectory_id + 1}")
    
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
    
    print(f"🚀 Manual control active. Use keyboard to control the robot!")
    
    while not done:
        # Display current observation
        key = display_observation(obs)
        
        # Handle special keys
        if key == 27:  # ESC - end trajectory
            print("🛑 Trajectory ended by user")
            break
        elif key == ord('q'):  # Quit
            print("🛑 Quitting demo recording")
            return None
        elif key == ord('r'):  # Reset
            print("🔄 Resetting environment")
            obs, info = env.reset()
            continue
        elif key == ord('s'):  # Save trajectory (but continue)
            print("💾 Marking trajectory for save")
            
        # Get action from keyboard input
        action = controller.get_action_from_key(key)
        
        if action is None:  # Skip if no valid action
            continue
            
        # Store current observation
        trajectory['observations'].append(obs)
        
        # Take environment step
        print(f"Step {step_count}: Action {action}")
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
        
        # Check for success or max steps
        if reward > 0:  # Assuming positive reward indicates progress
            print(f"🎉 Good action! Reward: {reward}")
            
        if step_count >= 1000:  # Max episode length
            print("⏰ Maximum episode length reached")
            break
    
    cv2.destroyAllWindows()
    
    if len(trajectory['actions']) > 0:
        print(f"✅ Trajectory {trajectory_id + 1} completed!")
        print(f"   Steps: {step_count}, Total reward: {total_reward:.2f}")
        
        # Convert lists to numpy arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        
        return trajectory
    else:
        print(f"❌ Trajectory {trajectory_id + 1} was empty")
        return None

def record_demonstrations_interactive(env_name: str = 'MobileRobotEnv-v0', 
                                    num_demos: int = 20, 
                                    save_path: str = 'mobile_robot_demos.pkl'):
    """
    Record demonstration trajectories with interactive manual control.
    """
    print(f"🎯 Interactive demo recording for {env_name}")
    print(f"Target: {num_demos} demonstrations")
    
    # Create environment
    env = gym.make(env_name)
    
    # Apply transformations
    def transform_obs(obs):
        new_obs = obs.copy()
        if 'rgb' in new_obs:
            new_obs['front'] = new_obs.pop('rgb')
        return new_obs
    
    env = TransformObservation(env, transform_obs)
    
    # Create controller
    controller = ManualController(env.action_space)
    
    trajectories = []
    trajectory_id = 0
    
    print(f"\n🎮 Starting interactive demo recording...")
    print(f"📝 Record {num_demos} successful banana grasping demonstrations")
    
    while len(trajectories) < num_demos:
        try:
            trajectory = record_manual_demo(env, trajectory_id, controller)
            
            if trajectory is not None:
                # Ask user if this trajectory should be saved
                print(f"\n📊 Trajectory {trajectory_id + 1} stats:")
                print(f"   Steps: {len(trajectory['actions'])}")
                print(f"   Total reward: {np.sum(trajectory['rewards']):.2f}")
                
                save_traj = input("💾 Save this trajectory? (y/n): ").lower()
                if save_traj in ['y', 'yes']:
                    trajectories.append(trajectory)
                    print(f"✅ Saved! Progress: {len(trajectories)}/{num_demos}")
                else:
                    print("🗑️  Trajectory discarded")
            
            trajectory_id += 1
            
            if len(trajectories) < num_demos:
                cont = input(f"\n🔄 Continue recording? ({len(trajectories)}/{num_demos} completed) (y/n): ").lower()
                if cont in ['n', 'no', 'q', 'quit']:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⏹️  Recording interrupted. Saved {len(trajectories)} trajectories.")
            break
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            continue
    
    env.close()
    cv2.destroyAllWindows()
    
    if trajectories:
        # Save trajectories
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(trajectories, f)
        
        print(f"\n🎉 Successfully recorded and saved {len(trajectories)} demonstrations!")
        print(f"💾 Saved to: {save_path}")
        
        # Print statistics
        total_steps = sum(len(traj['actions']) for traj in trajectories)
        avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories])
        
        print(f"\n📊 Final Statistics:")
        print(f"   Trajectories: {len(trajectories)}")
        print(f"   Total steps: {total_steps}")
        print(f"   Average reward: {avg_reward:.2f}")
        
        print(f"\n🚀 Ready for training! Use:")
        print(f"   bash run_learner.sh --demo_path {save_path}")
        
        return save_path
    else:
        print("❌ No demonstrations recorded.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Record interactive demonstrations for mobile robot")
    parser.add_argument("--env", default="MobileRobotEnv-v0", help="Environment name")
    parser.add_argument("--num_demos", type=int, default=20, help="Number of demo trajectories to record")
    parser.add_argument("--save_path", default="mobile_robot_demos.pkl", help="Path to save demo trajectories")
    
    args = parser.parse_args()
    
    demo_path = record_demonstrations_interactive(
        env_name=args.env,
        num_demos=args.num_demos,
        save_path=args.save_path
    )
