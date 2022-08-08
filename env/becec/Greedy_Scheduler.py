import numpy as np
import time

from .Environment import Environment
from .Task import Task
from utils.logger import Logger

class Scheduler():
    def __init__(self, config):
        self.config = config
        self._env = Environment(config=config)

        if self._env.is_local_file_exsisted():
            self._env.loadEnv()
            print(f"Loaded environment. Steady distribution of BS 0 is {self._env.BS[0].mmpp.steady_dist}")
        else:
            self._env.saveEnv()
            print(f"Saved environment. Steady distribution of BS 0 is {self._env.BS[0].mmpp.steady_dist}")

        log_path = f"results/test/greedy"
        self.logger = Logger(log_path)

    def seed(self, seed):
        self._env.seed(seed)
    
    def close(self):
        self.reset()

    def go_next_frame(self):
        while True:
            self._env.next()
            if self._env.is_end_of_frame():
                break

    def reset(self):
        self._env.reset()
        self.go_next_frame()
    
    def step(self):
        M = self.config['M']
        delta_t = self.config['delta_t']
        env = self._env

        def find_allocation(task: Task, BS: int, t: int):
            # return: alloc_list, reward
            # 枚举从 t 开始的一个连续时间片, 该时间片长度由空闲资源与任务大小决定
            # 时间片不能超出 delta_t 范围
            c = env.C(BS, t)
            r = task.cpu_requirement()
            temp_t = t
            alloc_list = [0 for _ in range(delta_t)]
            while c < r and temp_t < delta_t - 1:
                temp_t += 1
                c += env.C(BS, temp_t)
            if c < r:
                return alloc_list, -1e6    # [0, ..., 0], -1e6
            
            u = task.utility(env.timer + temp_t)
            c = 0.

            temp_t = t
            while r > 0:
                cap = env.C(BS, temp_t)     # 剩余资源
                allo = min(r, cap)          # 分配量
                alloc_list[temp_t] = allo
                c += env.p(BS, temp_t) * allo
                r -= cap
                temp_t += 1
            return alloc_list, u-c

        reward = 0.

        # 1. 枚举当前 frame 中的任务
        for task in self._env.task_set:
            # 1.1 寻找 task 最佳的 BS 与 Slot
            max_r = -1e6
            target_BS = 0
            alloc_list = [0 for _ in range(delta_t)]
            for BS in range(M):
                for t in range(delta_t):
                    temp_list, r = find_allocation(task, BS, t)
                    if r > max_r:
                        max_r = r
                        target_BS = BS
                        alloc_list = temp_list

            if max_r <= -1e6:
                # 无法分配
                continue

            # 1.2 分配任务
            # 分配给 BS
            self._env.schedule_task_to_BS(task=task, BS_ID=target_BS)
            # 指定 slot
            self._env.allocate_task_at_BS(task=task, BS_ID=target_BS, alloc_list=alloc_list)

            reward += max_r

        # 2. 环境更新到下一个 frame
        # 采用 Observation.py 一样的 frame
        if self.config['frame_mode'] == 0:
            if self._env.is_end_of_frame:
                self.go_next_frame()
            else:
                self._env.next_task_batch()
        elif self.config['frame_mode'] == 1:
            self.go_next_frame()
        
        done = (self._env.timer > self.config['T']-1)
        return reward, done

    def run(self):
        counts = 0 
        counts_max = 10
        while counts < counts_max:
            counts+=1
            episode_reward = 0.

            self._env.reset()

            ep_start_time = time.time()
            # 完成一个 episode
            done = False
            while not done:
                reward, done = self.step()
                episode_reward += reward

            self.logger.scalar_summary(f"greedy/episode_reward", episode_reward, counts)
            self.logger.scalar_summary(f"greedy/episode_timing", time.time() - ep_start_time, counts)

