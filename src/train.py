import yaml
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.envs.minigrid_env import MiniGridEnvWrapper
from src.models.actor_critic import ActorCritic
from src.agents.a2c_agent import A2CAgent


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def train():

    config = load_config()

    env_names = config["env"]["names"]

    envs = [MiniGridEnvWrapper(name) for name in env_names]

    state_dim = 5
    action_dim = 3

    model = ActorCritic(
        state_dim,
        action_dim,
        config["model"]["hidden_dim"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    agent = A2CAgent(
        model,
        optimizer,
        config["training"]["gamma"]
    )

    rewards = []

    for episode in tqdm(range(config["training"]["episodes"])):

        # randomly choose env
        env = random.choice(envs)

        state = env.reset()

        done = False
        episode_reward = 0

        while not done:

            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.update(log_prob, value, reward, next_state, done)

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    plt.plot(rewards)
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train()