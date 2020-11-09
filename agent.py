import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as optim
import random
import os
import time


'''
Do not use GPU model in the submission.
You can change the structure of the neural network as you like.
'''

class NETWORK(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network example
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(NETWORK, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "sign": [],
        }
        self.ptr = 0

    def __len__(self):
        return len(self.buffer['state'])

    def push(self, **transition):
        if len(self) < self.capacity:
            for key in transition:
                self.buffer[key].append(transition[key])
        else:
            for key in transition:
                self.buffer[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, size):
        idx = np.random.choice(len(self), size, replace=False)
        sample = {k: [v[i] for i in idx] for k, v in self.buffer.items()}
        return sample

class DQN(object):
    def __init__(
        self,
        state_dim=4, # state_dim = env.observation_space.shape[0]
        action_dim=2, # action_dim = env.action_space.n
        hidden_dim=16,
        buffer_capacity=10000,
        batch_size=64,
        alpha=0.99,
        beta=1e-3, # initial learning rate
        eps=1,
        eps_min=0.01,
        eps_decay=0.995,
        target_update=100,
        draw_plot=False,
        optim='adam',
    ):
        self.build_networks(state_dim, action_dim, hidden_dim)
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=beta)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.dqn.parameters(), lr=beta)
        elif optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.dqn.parameters(), lr=beta)
        else:
            raise ValueError
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.draw_plot=draw_plot
        self.step = 0

        # Key hyperparameters
        self.alpha = alpha  # discount factor
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_update = target_update

        # For visualiztion
        self.history = {
            "loss": [],
            "loss_smooth": [],
            "eps": [],
            "score": [], # accumulated reward within an episode
            "score_smooth": [],
        }
        self.score = 0
        self.plot_interval = 500

    def build_networks(self, state_dim, action_dim, hidden_dim):
        self.dqn = NETWORK(state_dim, action_dim, hidden_dim)
        self.dqn_target = NETWORK(state_dim, action_dim, hidden_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

    def select_action(self, states: np.ndarray) -> int:
        if np.random.rand() < self.eps:  # bug: not randn
            action = np.random.randint(self.action_dim)
        else:
            assert states.ndim == 1
            states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
            action = self.dqn(states).argmax(1).item()
        return action

    def policy(self, states: np.ndarray) -> int:
        assert states.ndim == 1
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        action = self.dqn(states).argmax(1).item()
        return action

    def get_target_q(self, rewards, next_states, signs):
        next_q = self.dqn_target(next_states).max(1)[0]
        target_q = rewards + self.alpha * next_q * (1 - signs)
        return target_q

    def train(self,s0,a0,r,s1,sign):
        self.buffer.push(state=s0, action=a0, reward=r, next_state=s1, sign=sign)
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer.sample(self.batch_size)
            states = torch.tensor(batch["state"], dtype=torch.float32)
            actions = torch.tensor(batch["action"]).unsqueeze(1)
            next_states = torch.tensor(batch["next_state"], dtype=torch.float32)
            rewards = torch.tensor(batch["reward"])
            signs = torch.tensor(batch["sign"])

            current_q = self.dqn(states).gather(1, actions).squeeze()
            target_q = self.get_target_q(rewards, next_states, signs)
            #loss = F.smooth_l1_loss(current_q, target_q)
            loss = F.mse_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update history: loss
            self.history['loss'].append(loss.item())
            self.history['loss_smooth'].append(
                np.mean(self.history['loss'][-50:])
            )

        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        if self.step % self.target_update == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        # update history
        self.score += r
        if sign:
            self.history['score'].append(self.score)
            self.history['score_smooth'].append(
                np.mean(self.history['score'][-10:])
            )
            self.score = 0
        self.history['eps'].append(self.eps)
        if self.draw_plot and self.step % self.plot_interval == 0:
            self._plot()
        self.step += 1

    def _plot(self):
        """Plot the training progresses."""
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

        clear_output(True)
        plt.figure(figsize=(20, 5))

        plt.subplot(131)
        plt.title('Accumulated rewards: %s' % (
                  np.mean(self.history['score'][-10:])))
        plt.plot(self.history['score'], alpha=0.5)
        plt.plot(self.history['score_smooth'])
        plt.xlabel('Episodes')

        plt.subplot(132)
        plt.title('Loss')
        plt.plot(self.history['loss'], alpha=0.5)
        plt.plot(self.history['loss_smooth'])
        plt.xlabel('Updating steps')

        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(self.history['eps'])

        plt.show()

class DDQN(DQN):
    def __init__(
        self,
        state_dim=4, # state_dim = env.observation_space.shape[0]
        action_dim=2, # action_dim = env.action_space.n
        hidden_dim=16,
        buffer_capacity=10000,
        batch_size=64,
        alpha=0.99,
        beta=1e-3, # initial learning rate
        eps=1,
        eps_min=0.01,
        eps_decay=0.995,
        target_update=100,
        draw_plot=False,
        optim='adam',
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            buffer_capacity=buffer_capacity, #50000,
            batch_size=batch_size,
            alpha=alpha,
            beta=beta,
            eps=eps,
            eps_min=eps_min,
            eps_decay=eps_decay,
            target_update=target_update,
            draw_plot=draw_plot,
            optim=optim,
        )

    def build_networks(self, state_dim, action_dim, hidden_dim):
        self.dqn = NETWORK(state_dim, action_dim, hidden_dim)
        self.dqn_target = NETWORK(state_dim, action_dim, hidden_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        # NOTE: don't set self.dqn_target.eval()
        # Actually it doesn't matter. DDQN is different from double q learning.
        # The target network only updates by cloning from the primal network periodically.

    def get_target_q(self, rewards, next_states, signs):
        next_actions = self.dqn(next_states).argmax(1, keepdims=True)
        next_q = self.dqn_target(next_states).gather(1, next_actions).squeeze()
        target_q = rewards + self.alpha * next_q * (1 - signs)
        return target_q