import time
from copy import deepcopy
import numpy as np
from ai.dqn.dqn3d import DQNAgent
from ai.base_agent import BaseAgent
from game_env.base_env import BaseEnv

def start_train(params):
    import importlib
    # Get module and model class
    module_name = params["env.module_name"]
    class_name = params["env.class_name"]
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)
    print(f"\n## from {module_name} import {class_name} Success.")
    env = env_class()
    print(f"## Build env({env}) Success.")

    agent = DQNAgent(env.state_shape, env.num_actions, params)
    oppo_agent = deepcopy(agent)
    oppo_agent.set_eval_mode()  # 对手模型不训练
    start_time = time.time()
    for episode in range(params["train.num_episode"]):
        s1 = env.restart()
        loss_dict = {}
        episode_steps = 0
        episode_reward = 0
        episode_actions = []  # 调试用
        agent.set_train_mode()
        for t in range(params["train.max_episode_step"]):
            episode_steps += 1
            a = agent.get_action(s1)
            episode_actions.append(a)
            s2, r, done = env.step(a)
            episode_reward += r
            loss_dict = agent.learn(s1, a, r, s2, done)
            s1 = s2
            if done:
                break
        if episode % params["train.test_per_episode"] == 0:
            # TODO: 和基线AI对比
            # test_info = start_test(agent, oppo_agent, deepcopy(env), params)
            pass
        if episode % params["train.selfplay_update_per_episode"] == 0:
            print(f"trained episodes: {episode}, start test")
            # TODO: 自博弈模型更新策略
            # 先无脑更新对手模型
            test_info = start_test(agent, oppo_agent, deepcopy(env), params)
            # if test_info["win_rate1"] > 0.9:
            #     oppo_agent = deepcopy(agent)
            #     oppo_agent.set_eval_mode()
            for k, v in test_info.items():
                print(f"{k}: {v}", end=", ")
            print()

        if episode % 100 == 0:
            end_time = time.time()
            print(f"Episode: {episode}, steps: {episode_steps}, reward: {episode_reward}, time: {end_time-start_time}s")
            print(f"actions: {episode_actions}")
            start_time = end_time
            for k, v in loss_dict.items():
                print(f"{k}: {v}", end=", ")
            print()


def start_test(agent1: BaseAgent, agent2: BaseAgent, env: BaseEnv, params):
    agent1.set_eval_mode()
    agent2.set_eval_mode()
    start = time.time()
    l_episode_steps = []
    l_episode_reward = []
    for episode in range(params["test.num_episode"]):
        s1 = env.restart()
        episode_steps = 0
        episode_reward = 0
        episode_actions = []
        for t in range(params["test.max_episode_step"]):
            episode_steps += 1
            if s1["cur_player"] == 1:
                a = agent1.get_action(s1)
            else:
                a = agent2.get_action(s1)
            episode_actions.append(a)
            s2, r, done = env.step(a)
            episode_reward += r
            s1 = s2
            if done:
                break
        l_episode_steps.append(episode_steps)
        l_episode_reward.append(episode_reward)
    avg_episode_steps = np.mean(l_episode_steps)
    avg_episode_rewards = np.mean(l_episode_reward)
    win_rate1 = np.mean(np.where(np.greater(l_episode_reward, 0), 1.0, 0.0))
    win_rate2 = np.mean(np.where(np.less(l_episode_reward, 0), 1.0, 0.0))
    return {"steps": avg_episode_steps, "win_rate1": win_rate1, "win_rate2": win_rate2, "time": time.time()-start}

if __name__ == '__main__':
    import os
    from utils.parse_conf import ParseConf

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='conf.yaml', type=str)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.conf)
    params = ParseConf(config_path)
    start_train(params)
