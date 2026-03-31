import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BlockstackEnv(gym.Env):
    def __init__(self, N: int, N_max: int = 10000):
        super(BlockstackEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Box(shape=(1,), dtype=np.float32, low=-N**0.5-0.5, high=N**0.5-0.5)
        self.N_max = N_max
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(
        low=-1e6,
        high=1e6,
        shape=(N_max, 2),
        dtype=np.float32
        )

        self.blocks = [
            (-1.0, 1.0),
        ]

        self.num_blocks = 0
        self.N = N

    def get_observation(self):
        obs = np.zeros((self.N_max, 2), dtype=np.float32)
        for i, (x, y) in enumerate(self.blocks):
            obs[i] = [x, y]
        return obs

    def reset(self):
        # Reset the state of the environment to an initial state
        v = np.zeros((2, self.N_max), dtype=np.float32)
        return v

    def _compute_level(self, action):
        x = action[0]
        level = 0.0
        for (x_i, y_i) in self.blocks:
            if abs(x - x_i) < 1.0 and y_i > level:
                level = y_i

        return level + 1.0

    def step(self, action):

        # compute level (for now just placeholder)
        level = self._compute_level(action)

        self.blocks.append((action[0], level))
        self.num_blocks += 1


    def render(self):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Draw each block
        for (x, y) in self.blocks:
            rect = patches.Rectangle(
                (x, y),  # bottom-left corner
                1.0,                 # width
                1.0,                 # height
                edgecolor='black',
                facecolor='skyblue'
            )
            ax.add_patch(rect)

        # Set limits (adjust dynamically)
        if self.blocks:
            xs = [x for x, _ in self.blocks]
            ys = [y for _, y in self.blocks]

            ax.set_xlim(min(xs) - 3, max(xs) + 3)
            ax.set_ylim(0, max(ys) + 3)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 5)

        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

