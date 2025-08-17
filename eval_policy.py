import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from absl import app, flags
from flax.training import checkpoints

from serl_launcher.utils.launcher import make_drq_agent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import mobile_robot

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", "./checkpoints", "Path to saved checkpoints.")
flags.DEFINE_integer("checkpoint_step", None, "Specific checkpoint step to load (if None, loads latest).")
flags.DEFINE_string("env", "PiperMobileRobot-v0", "Name of environment.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_bool("render", True, "Whether to render the environment.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")

def main(_):
    # Create environment
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env)
    
    # Apply same transformations as training
    from gymnasium.wrappers import TransformObservation
    
    def transform_obs(obs):
        new_obs = obs.copy()
        if 'rgb' in new_obs:
            new_obs['front'] = new_obs.pop('rgb')
        return new_obs
    
    # Get the observation space after transformation
    sample_obs = env.observation_space.sample()
    transformed_sample = transform_obs(sample_obs)
    
    # Create new observation space
    from gymnasium import spaces
    new_obs_space = spaces.Dict({
        key: env.observation_space[orig_key] if key == 'state' 
             else env.observation_space['rgb'] if key == 'front'
             else space
        for key, space in [(k, env.observation_space[k] if k in env.observation_space else v) 
                          for k, v in {'front': None, 'state': None}.items()]
        for orig_key in ['rgb', 'state'] if orig_key in env.observation_space
    })
    
    # Properly construct the new observation space
    new_obs_space = spaces.Dict({
        'front': env.observation_space['rgb'],
        'state': env.observation_space['state']
    })
    
    env = TransformObservation(env, transform_obs, new_obs_space)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    # Create agent
    agent = make_drq_agent(
        seed=42,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # Load checkpoint
    if FLAGS.checkpoint_step:
        restored_state = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state, step=FLAGS.checkpoint_step
        )
    else:
        # Load latest checkpoint
        restored_state = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state
        )
    
    agent = agent.replace(state=restored_state)
    
    print(f"Loaded checkpoint from {FLAGS.checkpoint_path}")
    
    # Run evaluation episodes
    total_returns = []
    success_count = 0
    
    for episode in range(FLAGS.num_episodes):
        obs, _ = env.reset()
        total_return = 0
        step_count = 0
        done = False
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step_count < 1000:  # Max 1000 steps per episode
            # Sample action from policy (deterministic for evaluation)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=True,  # Deterministic policy
            )
            actions = np.asarray(jax.device_get(actions))
            
            # Step environment
            obs, reward, done, truncated, info = env.step(actions)
            total_return += reward
            step_count += 1
            
            if done or truncated:
                break
        
        # Check if episode was successful using proper criterion
        episode_success = info.get('is_success', False)  # Use environment's success flag
        
        total_returns.append(total_return)
        if episode_success:  # Use proper success criterion instead of positive reward
            success_count += 1
            
        print(f"  Return: {total_return:.2f}, Steps: {step_count}, Success: {episode_success}")
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({FLAGS.num_episodes} episodes):")
    print(f"Mean Return: {np.mean(total_returns):.2f} ± {np.std(total_returns):.2f}")
    print(f"Success Rate: {success_count}/{FLAGS.num_episodes} ({100*success_count/FLAGS.num_episodes:.1f}%)")
    print(f"{'='*50}")

if __name__ == "__main__":
    app.run(main)
