import gym

def eval_policy(agent,env_name,eval_episodes=10):
    eval_env = gym.make(env_name)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.policy(state)
            state, reward, done,_ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
#     print("---------------------------------------")
    return avg_reward

'''Return an episode with a given policy'''

def return_episode(agent,env_name):
    gen_env = gym.make(env_name)
    state_list = []
    action_list = []
    reward_list = []
    done_list = []

    state, done = gen_env.reset(), False
    state_list.append(state)
    episode_timesteps = 0
    while not done:
        episode_timesteps += 1
        action = agent.select_action(state)
        state, reward, done,_ = gen_env.step(action)

        action_list.append(action)
        state_list.append(state)
        reward_list.append(reward)
        if done:
            done_list.append(1)
        else:
            done_list.append(0)
    return state_list,action_list,reward_list,done_list