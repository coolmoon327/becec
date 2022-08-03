from .env_wrapper import EnvWrapper
import numpy as np


class BECEC(EnvWrapper):
    def __init__(self, config):
        EnvWrapper.__init__(self, config['env'])
        self.config = config
        self.env_name = config['env']
        self.env.load_config(config)

        # 用来输出上下限，方便调整 D4PG 的上下限
        self.config['log_min'] = 1e5
        self.config['log_max'] = -1e5

        # self.rewards = []
        self.mean_reward = 0.
        self.counts = 0

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        if self.config['log_max'] < reward:
            self.config['log_max'] = reward
            print(f"Max reward: {self.config['log_max']}")
        if self.config['log_min'] > reward:
            self.config['log_min'] = reward
            print(f"Min reward: {self.config['log_min']}")

        # return reward

        # self.rewards.append(reward)
        # Mean = np.mean(self.rewards)
        # Std = np.std(self.rewards)
        # if Std==0:
        #     Std = 1.
        # norm_reward = (reward - Mean) / Std

        norm_reward = reward - self.mean_reward
        self.counts += 1
        if self.counts == 1:
            self.mean_reward = reward
        else:
            Gamma = 1./self.counts
            Gamma = max(Gamma, 1e-3)
            self.mean_reward = (1-Gamma)*self.mean_reward + Gamma*reward
        
        return norm_reward
    
    def get_details(self):
        details = self.env.get_details()
        return details

    @classmethod
    def set_config(cls, config):
        state_mode = config['state_mode']
        action_mode = config['action_mode']
        M = config['M']
        delta_t = config['delta_t']
        n_tasks = config['n_tasks']
        
        if state_mode == 0:
            # 方案一 - 观测内容：基站信息（delta_t 个时隙的 C 和 p） + 任务信息（w, u_0, alpha）
            n_observations = M*delta_t*2 + n_tasks*2
        elif state_mode == 1:
            # 方案二 - 观测内容：基站信息（C、p 信息聚合到两个维度上） + 任务信息（w, u_0, alpha）
            n_observations = M*2 + n_tasks*2         
        elif state_mode == 2:
            # 方案三 - 观测内容：基站信息（delta_t 个时隙的 C） + 任务信息（w, u_0, alpha）
            n_observations = M*delta_t + n_tasks*2

        if action_mode == 0:
            # 方案一 - 行为内容：任务调度（目标基站） 输出 [-1, 1] 量化成 M+1 个基站编号（其中一个是 null）
            n_actions = n_tasks
        elif action_mode == 1:
            # 方案二 - 行为内容：one-hot    输出 (M+1) 个 [-1, 1] 为一组，表示该任务调度到各基站上去的 one-hot 概率
            n_actions = n_tasks * (M+1)
        
        config['state_dim'] = n_observations
        config['action_dim'] = n_actions
        config['action_low'] = -1.
        config['action_high'] = 1.