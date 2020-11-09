import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from argparse import ArgumentParser

from agent import DQN, DDQN
from utils import eval_policy


def train(seed, model, beta, optim):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    if model == 'dqn':
        Model = DDQN
    elif model == 'ddqn':
        Model = DDQN
    agent = Model(beta=beta, optim=optim)
    with trange(300, leave=False) as t:
        for i in t:
            state, done = env.reset(), False
            episodic_reward = 0
            while not done:
                action = agent.select_action(np.squeeze(state))
                next_state, reward, done, info = env.step(action)
                episodic_reward += reward
                sign = 1 if done else 0
                agent.train(state, action, reward, next_state, sign)
                state = next_state
            t.set_description('Episode {}'.format(i))
            if len(agent.history['loss_smooth']) > 0:
                t.set_postfix(score=agent.history['score_smooth'][-1],
                              loss=agent.history['loss_smooth'][-1])
    history = {
        'loss': agent.history['loss'],
        'score': agent.history['score'],
        'eval_reward': eval_policy(agent,env_name,eval_episodes=50)
    }
    return history

def run_experiment(model, optim, beta):
    num_reps = 6
    history_reps = []
    for seed in trange(num_reps):
        history_reps.append(train(seed, model=model, beta=beta, optim=optim))
        np.save('reps_{}_{}'.format(optim, beta), history_reps)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', choices=['dqn', 'ddqn'],
                        help="Deep Q-learning model")
    parser.add_argument('optim', choices=['adagrad', 'rmsprop', 'adam'],
                        help="Optimizer type")
    parser.add_argument('beta', type=float,
                        help="Learning rate")
    args = parser.parse_args()

    run_experiment(args.model, args.optim, args.beta)
