import torch

from src.envs.minigrid_env import MiniGridEnvWrapper
from src.models.actor_critic import ActorCritic


def evaluate(model_path="model.pth", episodes=20):

    env = MiniGridEnvWrapper("BabyAI-GoToLocal-v0")

    model = ActorCritic(5, 3, 128)

    model.load_state_dict(torch.load(model_path))

    model.eval()

    success = 0

    for _ in range(episodes):

        state = env.reset()
        done = False

        while not done:

            state_t = torch.tensor(state).float().unsqueeze(0)

            logits, _ = model(state_t)

            action = torch.argmax(logits).item()

            state, reward, done, _ = env.step(action)

            if reward > 0:
                success += 1

    print("Success rate:", success / episodes)


if __name__ == "__main__":
    evaluate()