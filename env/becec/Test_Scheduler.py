from multiprocessing.sharedctypes import Value
import numpy as np
import time

from .Environment import Environment
from .Task import Task
from utils.logger import Logger

from .stage_two.Stage_Two_Pointer import Stage_Two_Pointer

class Scheduler():
    def __init__(self, config, mode=0):
        # mode: 
        # 0 - greedy
        # 1 - fully random
        # 2 - partial random
        # 10 - obtain auto-encoder trainning data
        self.mode = mode
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

        self.bs_observation_buffer = []

        self.alg_2 = Stage_Two_Pointer(self._env)

    def seed(self, seed):
        self._env.seed(seed)
    
    def close(self):
        self.reset()

    def get_observation(self, bs):
        delta_t = self.config['delta_t']
        state = [0. for _ in range(delta_t)]
        for t in range(delta_t):
            state[t] = self._env.C(bs, t) / 1e3
        return state

    def go_next_frame(self):
        while True:
            self._env.next()

            if self.mode == 10:
                # 每一步都记录一下 BS observation
                for bs in range(self.config['M']):
                    self.bs_observation_buffer.append(self.get_observation(bs))

            if self._env.is_end_of_frame():
                break

    def reset(self):
        self._env.reset()
        self.alg_2.reset()
        self.go_next_frame()

        self.target_BS.clear()
        self.thrown_num = 0
    
    def find_allocation(self, task: Task, BS: int, t: int):
        delta_t = self.config['delta_t']
        # return: alloc_list, reward
        # 枚举从 t 开始的一个连续时间片, 该时间片长度由空闲资源与任务大小决定
        # 时间片不能超出 delta_t 范围
        c = self._env.C(BS, t)
        r = task.cpu_requirement()
        temp_t = t
        alloc_list = [0 for _ in range(delta_t)]
        while c < r and temp_t < delta_t - 1:
            temp_t += 1
            c += self._env.C(BS, temp_t)
        if c < r:
            return alloc_list, -1e6    # [0, ..., 0], -1e6
        
        u = task.utility(self._env.timer + temp_t)
        cost = 0.
        temp_t = t
        while r > 0:
            cap = self._env.C(BS, temp_t)     # 剩余资源
            allo = min(r, cap)          # 分配量
            alloc_list[temp_t] = allo
            cost += self._env.p(BS, temp_t) * allo
            r -= cap
            temp_t += 1
        return alloc_list, u-cost

    def find_allocation_only_at_t(self, task: Task, BS: int, t: int):
        c = self._env.C(BS, t)
        r = task.cpu_requirement()
        alloc_list = [0 for _ in range(self.config['delta_t'])]
        if c < r:
            return alloc_list, -1e6
        alloc_list[t] = r
        u = task.utility(self._env.timer + t)
        cost = self._env.p(BS, t) * r
        return alloc_list, u-cost

    def greedy_search(self, task):
        M = self.config['M']
        delta_t = self.config['delta_t']
        max_r = -1e6
        target_BS = -1
        alloc_list = [0 for _ in range(delta_t)]
        for BS in range(M):
            if self._env.is_resource_enough_in_BS(task, BS):
                for t in range(delta_t):
                    temp_list, r = self.find_allocation(task, BS, t)
                    # temp_list, r = find_allocation_only_at_t(task, BS, t)
                    if r > max_r:
                        max_r = r
                        target_BS = BS
                        alloc_list = temp_list

        return max_r, target_BS, alloc_list

    def random_search(self, task):
        M = self.config['M']
        delta_t = self.config['delta_t']
        BS = np.random.randint(0, M)
        t = np.random.randint(0, delta_t)
        
        # 纯随机
        alloc_list, r = self.find_allocation(task, BS, t)
        if r == -1e6:
            BS = -1
        return r, BS, alloc_list

    def partial_random_search(self, task):
        M = self.config['M']
        delta_t = self.config['delta_t']
        BS = np.random.randint(0, M)
        t = np.random.randint(0, delta_t)

        # 尽可能安排任务的随机
        for j in range(M):
            target_bs = (BS+j)%M
            if self._env.is_resource_enough_in_BS(task, target_bs):
                for i in range(delta_t):
                    target_t = (t-i)%delta_t    # t 越小越可能有效, 因此用 t-i
                    alloc_list, r = self.find_allocation(task, target_bs, target_t)
                    if r > -1e6:
                        return r, target_bs, alloc_list
        return -1e6, -1, [0 for _ in range(delta_t)]

    def step(self):
        reward = 0.

        def search(task):
            if self.mode == 0:
                r, target_BS, alloc_list = self.greedy_search(task)
            elif self.mode == 1 or self.mode == 10:
                r, target_BS, alloc_list = self.random_search(task)
            elif self.mode == 2:
                r, target_BS, alloc_list = self.partial_random_search(task)
            else:
                raise ValueError(f"Invalid parameter mode={self.mode}")
            return r, target_BS, alloc_list

        # 1. 枚举当前 frame 中的任务
        if not self.config['test_with_stage2'] and self.mode!=10:
            # A 使用全局 test 算法
            for task in self._env.task_set:
                # 1.1 寻找可执行 task 的 BS 与 Slot
                r, target_BS, alloc_list = search(task)
                if target_BS == -1:
                    # 无法分配
                    self.thrown_num += 1
                    continue

                # 1.2 分配任务
                # 分配给 BS
                self._env.schedule_task_to_BS(task=task, BS_ID=target_BS)
                self.target_BS.append(target_BS)
                # 指定 slot
                self._env.allocate_task_at_BS(task=task, BS_ID=target_BS, alloc_list=alloc_list)
                reward += r

        else:
            # B 使用全局 test 算法找 BS, 使用二阶段算法分配 slots
            for task in self._env.task_set:
                # 1.1 寻找可执行 task 的 BS 与 Slot
                r, target_BS, alloc_list = search(task)
                if target_BS == -1:
                    # 无法分配
                    self.thrown_num += 1
                    continue
                # 分配给 BS
                self._env.schedule_task_to_BS(task=task, BS_ID=target_BS)
                self.target_BS.append(target_BS)

                # 这一部分如果在 for 之外执行, 就是和 RL 类似的分配方式, 但是会导致第一阶段的 greedy 失效
                # 具体原因是, global greedy 不会考虑缓存区里暂未分配的任务, 因此性能会降低很多
                # 1.2 分配任务
                c, u, penalty = self.alg_2.execute()
                r = u - c
                self.thrown_num += self.alg_2.get_thrown_tasks_num()
                reward += r

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

    def save(self, checkpoint_name):
        import os
        process_dir = "./results/Auto_Encoder"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        filename = f"{process_dir}/{checkpoint_name}.npy"
        np.save(filename, np.array(self.bs_observation_buffer))

    def run(self):
        counts = 0 
        if self.mode == 10:
            counts_max = self.config['ae_pretrain_episodes_count']
        else:
            counts_max = self.config['test_episodes_count']

        sum_ = np.zeros(3)
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

            if self.mode == 10:
                # mode 10 不记录 log
                if counts%100==0:
                    print(f"Episode {counts}, got exp buffer with size {len(self.bs_observation_buffer)}")
                continue

            episode_time = time.time() - ep_start_time
            self.logger.scalar_summary(f"greedy/episode_reward", episode_reward, counts)
            self.logger.scalar_summary(f"greedy/thrown_tasks", self.thrown_num, counts)
            self.logger.scalar_summary(f"greedy/episode_timing", episode_time, counts)
            print(f"---\nEpisode {counts}")
            BS_print = []
            for BS in range(self.config["M"]):
                BS_print.append(f"BS{BS}: {self.target_BS.count(BS)}")
            
            sum_[0] += self.thrown_num
            sum_[1] += episode_reward
            sum_[2] += episode_time
            print(f"{BS_print}\nThrown tasks: {self.thrown_num} | Pure reward: {episode_reward}")
            print("---")
        
        if self.mode != 10:
            print(f"Mean: Thrown tasks: {sum_[0]/counts_max} | Pure reward: {sum_[1]/counts_max} | Episode Time: {sum_[2]/counts_max}")
        else:
            self.save(f"database_dt_{self.config['delta_t']}")
            print("Finish observation data gaining.")
