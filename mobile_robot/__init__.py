from gymnasium.envs.registration import register
from mobile_robot.mobile_robot_env import PiperEnv

# Register the mobile robot environment
register(
    id='PiperMobileRobot-v0',
    entry_point='mobile_robot.mobile_robot_env:PiperEnv',
    max_episode_steps=80,
)

__all__ = ['PiperEnv']
