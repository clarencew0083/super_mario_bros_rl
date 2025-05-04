import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

import gym_compatibility_wrapper
from nes_py.wrappers import JoypadSpace
import wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY

import agent
import utility


if __name__ == "__main__":
    raw_env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)

    # Wrap the raw gym environment in our Gymnasium-compatible wrapper.
    env = gym_compatibility_wrapper.GymCompatibilityWrapper(raw_env)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrapper.MaxAndSkipEnv(env)
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # Stack 4 frames

    # adjust number of episodes as desired
    episodes = 4
    batch_size = 64 # Set a batch size
    scores = []
    eps_history = []

    agent = agent.Agent(input_dims=env.observation_space.shape,
                n_actions=env.action_space.n, lr=5e-5)


    utility.run_episode(episodes, env, agent, scores,  eps_history, batch_size)