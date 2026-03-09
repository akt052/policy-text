import torch
import torch.nn.functional as F


class A2CAgent:

    def __init__(self, model, optimizer, gamma):

        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state):

        state = torch.tensor(state).float().unsqueeze(0)

        logits, value = self.model(state)

        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def update(self, log_prob, value, reward, next_state, done):

        next_state = torch.tensor(next_state).float().unsqueeze(0)

        with torch.no_grad():
            _, next_value = self.model(next_state)

        target = reward + self.gamma * next_value * (1 - done)

        advantage = target - value

        actor_loss = -log_prob * advantage.detach()

        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()