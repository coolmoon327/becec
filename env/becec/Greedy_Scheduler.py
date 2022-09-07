import numpy as np
import time

from .Environment import Environment
from .Task import Task
from utils.logger import Logger

from .stage_two.Stage_Two_Pointer import Stage_Two_Pointer

# 是否使用第二阶段算法 (若是, 则只会将任务发送给 BS, 由第二阶段执行分配)
use_alg_2 = False

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

        self.target_BS = []
        self.thrown_num = 0

        self.alg_2 = Stage_Two_Pointer(self._env)

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
        self.alg_2.reset()
        self.go_next_frame()

        self.target_BS.clear()
        self.thrown_num = 0
    
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

        def find_allocation_only_at_t(task: Task, BS: int, t: int):
            c = env.C(BS, t)
            r = task.cpu_requirement()
            alloc_list = [0 for _ in range(delta_t)]
            if c < r:
                return alloc_list, -1e6
            alloc_list[t] = r
            u = task.utility(env.timer + t)
            c = env.p(BS, t) * r
            return alloc_list, u-c

        reward = 0.

        # 1. 枚举当前 frame 中的任务
        for task in self._env.task_set:
            # 1.1 寻找 task 最佳的 BS 与 Slot
            max_r = -1e6
            target_BS = -1
            alloc_list = [0 for _ in range(delta_t)]
            for BS in range(M):
                for t in range(delta_t):
                    temp_list, r = find_allocation(task, BS, t)
                    # temp_list, r = find_allocation_only_at_t(task, BS, t)
                    if r > max_r:
                        max_r = r
                        target_BS = BS
                        alloc_list = temp_list

            if target_BS == -1:
                # 无法分配
                self.thrown_num += 1
                continue

            left_source = 0
            for t in range(delta_t):
                left_source += self._env.C(target_BS, t)
            a = left_source

            # 1.2 分配任务
            # 分配给 BS
            self._env.schedule_task_to_BS(task=task, BS_ID=target_BS)
            if use_alg_2:
                c, u, penalty = self.alg_2.execute()
                r = u - c + penalty
            else:
                # 指定 slot
                self._env.allocate_task_at_BS(task=task, BS_ID=target_BS, alloc_list=alloc_list)
                r = max_r
            reward += r

            left_source = 0
            for t in range(delta_t):
                left_source += self._env.C(target_BS, t)
            b = left_source
            # print(f"{a} - {b} = {task.cpu_requirement()}")

            self.target_BS.append(target_BS)

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
        counts_max = 100
        while counts < counts_max:
            counts+=1
            episode_reward = 0.

            self.reset()

            ep_start_time = time.time()
            # 完成一个 episode
            done = False
            while not done:
                reward, done = self.step()
                episode_reward += reward

            self.logger.scalar_summary(f"greedy/episode_reward", episode_reward, counts)
            self.logger.scalar_summary(f"greedy/episode_timing", time.time() - ep_start_time, counts)
            print(f"---\nEpisode {counts}")
            BS_print = []
            for BS in range(self.config["M"]):
                BS_print.append(f"BS{BS}: {self.target_BS.count(BS)}")
            print(f"{BS_print}\nThrown tasks: {self.thrown_num} | Pure reward: {episode_reward}")
            print("---")
