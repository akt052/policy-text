import gymnasium as gym
import minigrid
import numpy as np

from src.utils.preprocess import parse_mission, encode_object


class MiniGridEnvWrapper:

    def __init__(self, env_name):

        self.env = gym.make(env_name)

    def reset(self):

        obs, _ = self.env.reset()

        return self._build_state(obs)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        state = self._build_state(obs)

        return state, reward, done, info

    def _build_state(self, obs):

        env = self.env.unwrapped

        agent_x, agent_y = env.agent_pos
        agent_dir = env.agent_dir

        mission = obs["mission"]

        color, obj = parse_mission(mission)

        obj_id, color_id = encode_object(obj, color)

        target_x, target_y = self._find_target(obj, color)

        dx = target_x - agent_x
        dy = target_y - agent_y

        state = np.array(
            [
                dx,
                dy,
                agent_dir,
                obj_id,
                color_id
            ],
            dtype=np.float32
        )

        return state

    def _find_target(self, obj_type, color):

        grid = self.env.unwrapped.grid

        for x in range(grid.width):
            for y in range(grid.height):

                cell = grid.get(x, y)

                if cell is None:
                    continue

                if cell.type == obj_type and cell.color == color:
                    return x, y

        return 0, 0