import numpy

from env.becec.Environment import Environment
import numpy as np
import time
import torch
from gym import spaces


# TODO 添加一种 state 模式，不记录具体的 p，只记录 p_coef

class Observation(object):
    def __init__(self, config):
        self.config = config
        self._env = Environment(config=config)
        
        # TODO check whether it works while paralleling
        if self.env.is_local_file_exsisted():
            self.env.loadEnv()
            print(self.env.BS[0].mmpp.steady_dist)
        else:
            self.env.saveEnv()
            print(self.env.BS[0].mmpp.steady_dist)
        
        state_mode = config['state_mode']
        action_mode = config['action_mode']
        M = config['M']
        delta_t = config['delta_t']
        n_tasks = config['n_tasks']
                
        if state_mode == 0:
            # 方案一 - 观测内容：基站信息（delta_t 个时隙的 C 和 p） + 任务信息（w, u_0, alpha）
            self.n_observations = M*delta_t*2 + n_tasks*3     
        elif state_mode == 1:
            # 方案二 - 观测内容：基站信息（C、p 信息聚合到两个维度上） + 任务信息（w, u_0, alpha）
            self.n_observations = M*2 + n_tasks*3               
        
        if action_mode == 0:
            # 方案一 - 行为内容：任务调度（目标基站） 输出 [-1, 1] 量化成 M+1 个基站编号（其中一个是 null）
            self.n_actions = n_tasks
        elif action_mode == 1:
            # 方案二 - 行为内容：one-hot    输出 (M+1) 个 [-1, 1] 为一组，表示该任务调度到各基站上去的 one-hot 概率
            self.n_actions = n_tasks * (M+1)

        self.action_space = spaces.Box(low=0, high=M-1, shape=(self.n_actions,))

    def _sort_batch_tasks(self):
        """
        对当前批的任务按照某种规则排序，以提升训练效果
        该方法会直接作用于 env 的 task_set，需要注意是否有其他方法对 task_set 的有序性有要求
        :return:
        """
        n_tasks = self.config['n_tasks']
        env = self._env
        batch_begin_index = env.task_batch_num * n_tasks  # 批首任务的下标
        next_batch_begin_index = min((env.task_batch_num+1) * n_tasks, len(env.task_set))  # 下一批首任务的下标
        temp_list = env.task_set[batch_begin_index:next_batch_begin_index]     # 取 [begin, next_begin) 之间的部分

        def sort_priority(elem):
            return -elem.w
        temp_list.sort(key=sort_priority)
        # 需要验证是否正确
        env.task_set[batch_begin_index:next_batch_begin_index] = temp_list

    def get_state(self):
        M = self.config['M']
        delta_t = self.config['delta_t']
        state_mode = self.config['state_mode']
        n_tasks = self.config['n_tasks']
        env = self._env
        
        state = [0. for _ in range(self.n_observations)]

        index = 0
        # 基站信息
        for i in range(M):
            # 该方案需要配合修改 parameter 的 n_observations 属性
            
            if state_mode == 0:
            # 方案一：直接把 delta_t 个 slots 的 c 和 p 记录下来作为状态
            # n_observations = M*delta_t*2 + n_tasks*3
                for t in range(delta_t):
                    state[index] = env.C(i, t) / 1e4
                    if env.p(i, t) < 1. :
                        state[index+1] = env.p(i, t) * 100
                    else:
                        state[index+1] = env.p(i, t)
                    index += 2
            
            elif state_mode == 1:
                # 方案二： delta_t 个 c 直接求和合并成一维， delta_t 个 p 进行反向 discounted 后，取倒数乘上 c 并求和成为一维
                #  n_observations = M*2 + n_tasks*3
                s1 = 0.
                s2 = 0.
                gamma = 0.99
                for t in range(delta_t):
                    c = env.C(i, t) / (4. * 1e4)
                    if env.p(i, t) < 1. :
                        p = env.p(i, t) * 100.
                    else:
                        p = env.p(i, t)
                    s1 += c
                    s2 = s2/gamma + c/p
                state[index] = s1
                state[index+1] = s2
                index += 2
            
        # 任务信息
        self._sort_batch_tasks()
        # 如果任务未满一批，剩下的全为 0.
        batch_begin_index = env.task_batch_num * n_tasks  # 批首任务的下标
        next_batch_begin_index = min((env.task_batch_num+1) * n_tasks, len(env.task_set))    # 下一批首任务的下标
        for n in range(batch_begin_index, next_batch_begin_index):
            state[index] = env.task_set[n].w / 10.
            state[index+1] = env.task_set[n].u_0 / 100.
            state[index+2] = env.task_set[n].alpha / 10.
            index += 3
        return numpy.array(state)

    def execute(self, action):
        """
        基于 action 将任务用 env.schedule_task_to_BS 交付给 BS
        action = 任务调度（目标基站 + 时隙）        len = n_tasks*2
        :param action:
        :return:
        """
        M = self.config['M']
        n_tasks = self.config['n_tasks']
        env = self._env
        
        batch_begin_index = env.task_batch_num * n_tasks  # 批首任务的下标
        next_batch_begin_index = min((env.task_batch_num+1) * n_tasks, len(env.task_set))  # 下一批首任务的下标
        # 这种调度没有管补 0 的那部分决策（action 后面可能还有一部分，但是 n 已经取到上限了）
        index = 0
        log_BS = []
        for n in range(batch_begin_index, next_batch_begin_index):
            task = env.task_set[n]
            target_BS = int(round(action[index]))
            index += 1
            if not 0 <= target_BS <= M:
                print(f"BS number {target_BS} is out of range!")
                # 修剪范围
                target_BS = min(M, target_BS)
                target_BS = max(0, target_BS)
            if target_BS == M:
                # null bs
                log_BS.append(-1)
            else:
                log_BS.append(target_BS)
                env.schedule_task_to_BS(task=task, BS_ID=target_BS)
        print(f"Target BS in stage one: {log_BS}")
        
    # 通常 DRL 的 env 设计，已弃用
    # def get_reward(self) -> int:
    #     # reward 需要使用 stage 2 获得
    #     pass
    #
    # def reset(self):
    #     print("Cannot reset environment!")
    #
    # def step(self, action):
    #     # 执行第一阶段
    #     # 1. 基于 action 将任务用 env.schedule_task_to_BS 交付给 BS
    #     pass
    #     # 2. 执行第二阶段（将第二阶段的算法当作一个黑盒模块）
    #     pass
    #     s_ = self.get_state()
    #     reward = self.get_reward()
    #     done = False
    #     return s_, reward, done
