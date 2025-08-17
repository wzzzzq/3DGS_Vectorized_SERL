#!/usr/bin/env python3

import sys
import os
# Add the serl root directory to the Python path
serl_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(serl_root))

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import cv2
import os
import psutil  # For memory monitoring

from typing import Any, Dict, Optional
import pickle as pkl
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import mobile_robot  # Import mobile robot package

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PiperMobileRobot-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("actor_queue_size", 100, "Queue size for actor data store (smaller values use less memory).")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")
flags.DEFINE_integer("num_envs", 1, "Number of parallel environments for vectorized data collection.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    Supports both single and vectorized environments.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # Create eval environment for mobile robot
    eval_env = gym.make(FLAGS.env)
    # Apply same transformations as training env
    from gymnasium.wrappers import TransformObservation
    import gymnasium.spaces as spaces
    
    def transform_obs(obs):
        new_obs = obs.copy()
        if 'rgb' in new_obs:
            new_obs['front'] = new_obs.pop('rgb')
        return new_obs
    
    # Create the transformed observation space for eval_env
    original_obs_space = eval_env.observation_space
    if isinstance(original_obs_space, spaces.Dict) and 'rgb' in original_obs_space.spaces:
        transformed_spaces = original_obs_space.spaces.copy()
        transformed_spaces['front'] = transformed_spaces.pop('rgb')
        transformed_obs_space = spaces.Dict(transformed_spaces)
    else:
        transformed_obs_space = original_obs_space
    
    eval_env = TransformObservation(eval_env, transform_obs, transformed_obs_space)
    eval_env = ChunkingWrapper(eval_env, obs_horizon=1, act_exec_horizon=None)
    eval_env = RecordEpisodeStatistics(eval_env)

    # Determine if we're using vectorized environments
    is_vectorized = hasattr(env, 'num_envs') and env.num_envs > 1
    num_envs = env.num_envs if is_vectorized else 1
    
    obs, _ = env.reset()
    
    # Initialize tracking variables for vectorized envs
    if is_vectorized:
        running_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=int)
    else:
        running_return = 0.0
        episode_length = 0

    # training loop
    timer = Timer()

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")
        
        # Monitor memory usage every 100 steps
        if step % 100 == 0:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Step {step}: Memory usage: {memory_mb:.1f} MB, Queue size: {len(data_store) if hasattr(data_store, '__len__') else 'unknown'}")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                if is_vectorized:
                    actions = np.array([env.single_action_space.sample() for _ in range(num_envs)])
                else:
                    actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                if is_vectorized:
                    # For vectorized environments, we need to sample actions for each environment individually
                    # The agent expects single observations, not batched ones
                    actions_list = []
                    for i in range(num_envs):
                        # Extract observation for single environment
                        single_obs = {}
                        for key_name, value in obs.items():
                            single_obs[key_name] = value[i]  # Take i-th environment's observation
                        
                        # Sample action for this single observation
                        single_action = agent.sample_actions(
                            observations=jax.device_put(single_obs),
                            seed=key,
                            deterministic=False,
                        )
                        actions_list.append(np.asarray(jax.device_get(single_action)))
                    
                    # Stack actions into batch
                    actions = np.stack(actions_list, axis=0)
                else:
                    # Single environment
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        deterministic=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            
            if is_vectorized:
                # Handle vectorized environment outputs
                reward = np.asarray(reward, dtype=np.float32)
                done = np.asarray(done, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool)
                terminated = done | truncated
                
                # Update running returns and episode lengths
                running_returns += reward
                episode_lengths += 1
                
                # Store transitions for each environment
                for env_idx in range(num_envs):
                    transition = dict(
                        observations={k: v[env_idx] for k, v in obs.items()} if isinstance(obs, dict) else obs[env_idx],
                        actions=actions[env_idx],
                        next_observations={k: v[env_idx] for k, v in next_obs.items()} if isinstance(next_obs, dict) else next_obs[env_idx],
                        rewards=reward[env_idx],
                        masks=1.0 - terminated[env_idx],
                        dones=terminated[env_idx],
                    )
                    data_store.insert(transition)
                
                # Reset tracking for terminated environments
                reset_mask = terminated
                running_returns[reset_mask] = 0.0
                episode_lengths[reset_mask] = 0
                
            else:
                # Handle single environment
                reward = np.asarray(reward, dtype=np.float32)
                info = np.asarray(info)
                running_return += reward
                episode_length += 1
                
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - (done or truncated),
                    dones=done or truncated,
                )
                data_store.insert(transition)

                if done or truncated:
                    running_return = 0.0
                    episode_length = 0

            obs = next_obs

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    rng,
    agent: DrQAgent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.num_envs > 1:
        # Create vectorized environment
        from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
        
        def make_single_env():
            if FLAGS.render:
                env = gym.make(FLAGS.env, render_mode="human")
            else:
                env = gym.make(FLAGS.env)
            return env
        
        # Use SyncVectorEnv for simplicity and reliability
        # You can switch to AsyncVectorEnv for potentially better performance
        env = SyncVectorEnv([make_single_env for _ in range(FLAGS.num_envs)])
        print_green(f"Created vectorized environment with {FLAGS.num_envs} parallel environments")
    else:
        # Single environment
        if FLAGS.render:
            env = gym.make(FLAGS.env, render_mode="human")
        else:
            env = gym.make(FLAGS.env)

    # Mobile robot environment already has dict observation space with 'rgb' and 'state'
    # We need to rename 'rgb' to match SERL convention and add chunking
    from gymnasium.wrappers import TransformObservation
    import gymnasium.spaces as spaces
    
    def transform_obs(obs):
        # Rename 'rgb' to 'front' to match SERL convention
        new_obs = obs.copy()
        if 'rgb' in new_obs:
            new_obs['front'] = new_obs.pop('rgb')
        return new_obs
    
    # For vectorized environments, we need special handling
    if FLAGS.num_envs > 1:
        # Create a comprehensive vectorized environment wrapper
        class CleanVectorizedEnvWrapper:
            def __init__(self, env):
                self.env = env
                self.num_envs = env.num_envs
                
                # Get reference spaces from a single environment
                single_env = gym.make(FLAGS.env)
                single_obs_space = single_env.observation_space
                single_action_space = single_env.action_space
                single_env.close()
                
                # Set up transformed observation space
                if isinstance(single_obs_space, spaces.Dict) and 'rgb' in single_obs_space.spaces:
                    transformed_spaces = single_obs_space.spaces.copy()
                    transformed_spaces['front'] = transformed_spaces.pop('rgb')
                    self.single_observation_space = spaces.Dict(transformed_spaces)
                else:
                    self.single_observation_space = single_obs_space
                
                # Create chunked observation space
                temp_env = gym.make(FLAGS.env)
                temp_env = TransformObservation(temp_env, transform_obs, self.single_observation_space)
                temp_env = ChunkingWrapper(temp_env, obs_horizon=1, act_exec_horizon=None)
                self.observation_space = temp_env.observation_space
                temp_env.close()
                
                # Action spaces
                self.single_action_space = single_action_space
                self.action_space = env.action_space
                
            def transform_obs(self, obs_dict):
                """Transform observations: rgb -> front."""
                if isinstance(obs_dict, dict) and 'rgb' in obs_dict:
                    new_obs = {}
                    for key, value in obs_dict.items():
                        if key == 'rgb':
                            new_obs['front'] = value
                        else:
                            new_obs[key] = value
                    return new_obs
                return obs_dict
                
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                transformed_obs = self.transform_obs(obs)
                # Apply chunking (pass through for obs_horizon=1)
                return transformed_obs, info
                
            def step(self, actions):
                obs, reward, done, truncated, info = self.env.step(actions)
                transformed_obs = self.transform_obs(obs)
                # Apply chunking (pass through for obs_horizon=1)
                return transformed_obs, reward, done, truncated, info
                
            def close(self):
                return self.env.close()
        
        env = CleanVectorizedEnvWrapper(env)
    else:
        # Single environment - use existing logic
        # Create the transformed observation space
        original_obs_space = env.observation_space
        if isinstance(original_obs_space, spaces.Dict) and 'rgb' in original_obs_space.spaces:
            transformed_spaces = original_obs_space.spaces.copy()
            transformed_spaces['front'] = transformed_spaces.pop('rgb')
            transformed_obs_space = spaces.Dict(transformed_spaces)
        else:
            transformed_obs_space = original_obs_space
        
        env = TransformObservation(env, transform_obs, transformed_obs_space)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    
    # For action space sampling, we need to handle vectorized vs single environments
    if FLAGS.num_envs > 1:
        # For vectorized environments, sample from the single action space
        sample_action = env.single_action_space.sample()
    else:
        sample_action = env.action_space.sample()
    
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=sample_action,
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )

        print_green("replay buffer created")
        print_green(f"replay_buffer size: {len(replay_buffer)}")

        # if demo data is provided, load it into the demo buffer
        # in the learner node, we support 2 ways to load demo data:
        # 1. load from pickle file; 2. load from tf rlds data
        if FLAGS.demo_path or FLAGS.preload_rlds_path:

            def preload_data_transform(data, metadata) -> Optional[Dict[str, Any]]:
                # NOTE: Create your own custom data transform function here if you
                # are loading this via with --preload_rlds_path with tf rlds data
                # This default does nothing
                return data

            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
                preload_rlds_path=FLAGS.preload_rlds_path,
                preload_data_transform=preload_data_transform,
            )

            if FLAGS.demo_path:
                # Check if the file exists
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                    for traj in trajs:
                        demo_buffer.insert(traj)

            print(f"demo buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,  # None if no demo data is provided
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(FLAGS.actor_queue_size)  # configurable queue size for actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
