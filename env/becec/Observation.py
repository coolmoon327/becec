import numpy
import numpy as np
import time
import torch
from gym import spaces

from .Environment import Environment
from .stage_two.Stage_Two_Pointer import Stage_Two_Pointer

# TODO 添加一种 state 模式，不记录具体的 p，只记录 p_coef

class Observation(object):
    def __init__(self, config={}):
        if len(config):
            self.load_config(config)
    
    def load_config(self, config):
        self.config = config
        self._env = Environment(config=config)
        
        self.alg_2 = Stage_Two_Pointer(self._env)
        
        # TODO check whether it works while paralleling
        if self._env.is_local_file_exsisted():
            self._env.loadEnv()
            print(self._env.BS[0].mmpp.steady_dist)
        else:
            self._env.saveEnv()
            print(self._env.BS[0].mmpp.steady_dist)
        
        self.n_observations = config['state_dim']
        self.n_actions = config['action_dim']
        
        # TODO 检查 action_space
        self.action_space = spaces.Box(low=config['action_low'], high=config['action_high'], shape=(self.n_actions,))
    
    def _sort_batch_tasks(self, batch_begin_index, next_batch_begin_index):
        """
        对当前批的任务按照某种规则排序，以提升训练效果
        该方法会直接作用于 env 的 task_set，需要注意是否有其他方法对 task_set 的有序性有要求
        :return:
        """
        env = self._env
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
        
        # 如果任务未满一批，剩下的全为 0.
        batch_begin_index = env.task_batch_num * n_tasks  # 批首任务的下标
        next_batch_begin_index = min((env.task_batch_num+1) * n_tasks, len(env.task_set))    # 下一批首任务的下标
        
        # 任务信息排序
        self._sort_batch_tasks(batch_begin_index, next_batch_begin_index)

        for n in range(batch_begin_index, next_batch_begin_index):
            state[index] = env.task_set[n].w / 10.
            state[index+1] = env.task_set[n].u_0 / 100.
            state[index+2] = env.task_set[n].alpha / 10.
            index += 3
        return numpy.array(state)

    def execute(self, action_raw):
        """
        基于 action 将任务用 env.schedule_task_to_BS 交付给 BS
        action = 任务调度（目标基站 + 时隙）        len = n_tasks*2
        :param action_raw:  未经处理的 actor 网络输出
        :return:
        """
        M = self.config['M']
        n_tasks = self.config['n_tasks']
        action_mode = self.config['action_mode']
        env = self._env
        
        # TODO 检查 actor 网络输出是否在 [-1., 1.]
        if type(action_raw) is np.ndarray:
            action_raw = np.clip(action_raw, -1., 1.)
        else:
            action_raw = action_raw[0, :].detach().clamp(-1., 1.).numpy()
        
        if action_mode == 0:
            # 方案一 - 在 [-1, 1] 上量化出选择的基站
            # 传入 action 的取值为 tanh，需要映射到 [0, M-1] 去
            action = action_raw + 1.        # [0., 2.]
            action = action * M/2.          # [0., M.]  第 M 个表示 null
            action = np.round(action)
        elif action_mode == 1:
            # 方案二 - 用类似 one hot 的方式，从每 (M+1) 个数中选择一个最大的，对应的下标就是选择的 BS
            n_tasks = n_tasks
            action = np.zeros(1, n_tasks)
            for i in range(n_tasks):
                first_index = i*(M+1)
                last_index = (i+1)*(M+1) - 1
                one_hot = action_raw[first_index:last_index+1]
                action[0][i] = np.argmax(one_hot)
        
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
    
    def seed(self, seed):
        self._env.seed(seed)
    
    def render(self, mode):
        # TODO 按照 render 的格式返回一个图
        pass
    
    def close(self):
        self.reset()
        # TODO more close opetation?
    
    def reset(self):
        self._env.reset()
        return self.get_state()
    
    def step(self, action):
        # 1. 执行第一阶段
        self.execute(action)
    
        # 2. 执行第二阶段（将第二阶段的算法当作一个黑盒模块）
        c, u = self.alg_2.execute()
        reward = u - c
    
        # 3. 环境更新到下一个 frame
        # TODO 实现按照到达任务数量的 frame 更新
        # TODO 验证是否会超出 T，以及有无影响
        while True:
            self._env.next()
            if self._env.is_end_of_frame():
                break
    
        s_ = self.get_state()
        
        # 更新到下一个 frame 后，如果 timer 溢出，说明 done
        # TODO done 所在的 frame 之后可能还有几个 slots，是否影响？
        done = (self._env.timer > self.config['T']-1)
        info = {}
        return s_, reward, done, info
