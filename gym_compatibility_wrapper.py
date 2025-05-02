from gymnasium import Env

class GymCompatibilityWrapper(Env):
    def __init__(self, env):
        self.env = env
        # Expose the same action and observation spaces:
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.render_mode = env.render_mode

    def reset(self, seed=None, options=None):
        # Depending on the underlying env's API you might need to adjust parameters.
        if seed is not None:
            return self.env.reset(seed=seed, options=options)
        else:
            return self.env.reset()

    def step(self, action):
        # Forward the step. Make sure the output is in the format expected by gymnasium.
        return self.env.step(action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def play(self, **kwargs):
        return self.env.play(**kwargs)

    def close(self):
        return self.env.close()