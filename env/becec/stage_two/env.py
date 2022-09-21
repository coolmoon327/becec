import torch
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import copy
import pickle


def get_2city_distance(n1, n2):
    x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
    if isinstance(n1, torch.Tensor):
        return torch.sqrt((x2 - x1).pow(2) + (y2 - y1).pow(2))
    elif isinstance(n1, (list, np.ndarray)):
        return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    else:
        raise TypeError


class Worker():
    def __init__(self, slot, task_number, utility):
        self.slot = slot
        self.task_number = task_number
        self.utility = utility


class Env_tsp():
    def __init__(self, cfg):
        '''
        nodes(cities) : contains nodes and their 2 dimensional coordinates
        [city_t, 2] = [3,2] dimension array e.g. [[0.5,0.7],[0.2,0.3],[0.4,0.1]]
        '''
        self.batch = cfg.batch
        self.city_t = cfg.city_t
        self.slots = cfg.slots  # 增加未来可使用的时隙数
        self.BS = 1

    def dp(self, slots_features, tasks):
        '''
        slots: ()
        return nodes:(city_t,2)
        '''
        pass

    def get_task_nodes(self, seed=None, get_env=None):
        '''
        return tasks:(city_t,2)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        '''
            BS_ID=self.BS
        '''

        self.slots = get_env.config['delta_t']

        task = get_env.get_BS_tasks_external(BS_ID=self.BS)
        data = np.empty((1, len(task), 2 + get_env.config['delta_t'] * 2))
        '''
            data 合成一批的数据
                1. 0 位置cpu_requirement
                2. 1 位置alpha
                3. 2 到 61 位置 env c 和 p
                    2 to 31 是 c
                    32 to 61 是 p
        '''
        self.city_t = len(task)
        for i in range(len(task)):
            data[0][i][0] = task[i].cpu_requirement()
            data[0][i][1] = task[i].alpha
            for slot in range(get_env.config['delta_t']):
                data[0][i][2 + slot] = \
                    get_env.C(i=self.BS, t=slot)  # i = self.BS
            for slot in range(get_env.config['delta_t']):
                # data[0][i][32 + slot] = \
                # 	get_env.p(i=self.BS, t=slot)
                data[0][i][2 + get_env.config['delta_t'] + slot] = get_env.p(
                    i=self.BS, t=slot)
        return torch.Tensor(data)

    def get_env_nodes(self, seed=None, get_env=None):
        '''
        return nodes:(n_samples,slots * 2)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        # env_computation_power = 10 * torch.rand((1, self.slots, 1)) + 5
        # env_price = 9 * torch.rand((1, self.slots, 1)) + 1 # fixme: env 参数名需要改变
        # env_info = torch.cat((env_computation_power, env_price), 2)
        # # reshape env to (batch, slots * 2)
        # env_info = torch.reshape(env_info, (1, self.slots * 2))
        # with open("Pkl/env-test-2022-03-02-11-09-20.pkl", "rb") as file:
        # 	env_info = torch.Tensor(pickle.load(file))
        envs = []
        for slot in range(get_env.config['delta_t']):
            envs.append(get_env.C(i=self.BS, t=slot))  # i = self.BS
        for slot in range(get_env.config['delta_t']):
            envs.append(get_env.p(i=self.BS, t=slot))  # i = self.BS
        return torch.Tensor([envs])  # 可以模仿的产生 通过 cat 拼接 两次产生的tensor

    def stack_nodes(self):
        '''
        nodes:(city_t,2)
        return inputs:(batch,city_t,2)
        '''
        list = [self.get_nodes() for i in range(self.batch)]
        inputs = torch.stack(list, dim=0)
        return inputs

    def get_batch_nodes(self, n_samples, seed=None):
        '''
        return nodes:(batch,city_t,2)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.rand((n_samples, self.city_t, 2), device=device)  # 可以模仿的产生

    def get_batch_task_nodes(self, n_samples, seed=None):
        '''
        return nodes:(batch,city_t,2)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        task_workload = 10 * torch.rand((n_samples, self.city_t, 1),
                                        device=device) + 5
        task_penalty_factor = torch.rand((n_samples, self.city_t, 1),
                                         device=device)  # fixme: task 参数名需要改变
        task_info = torch.cat((task_workload, task_penalty_factor), 2)
        return task_info  # 可以模仿的产生

    def get_batch_env_nodes(self, n_samples, seed=None):
        '''
        return nodes:(n_samples,slots * 2)
        '''
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        env_computation_power = 10 * torch.rand((n_samples, self.slots, 1),
                                                device=device) + 5
        env_price = 9 * torch.rand((n_samples, self.slots, 1),
                                   device=device) + 1  # fixme: env 参数名需要改变
        env_info = torch.cat((env_computation_power, env_price), 2)
        # reshape env to (batch, slots * 2)
        env_info = torch.reshape(env_info, (n_samples, self.slots * 2))
        return env_info  # 可以模仿的产生 通过 cat 拼接 两次产生的tensor

    def stack_random_tours(self, batch):
        '''
        tour:(city_t)
        return tours:(batch,city_t)
        '''
        list = [self.get_random_tour() for i in range(batch)]
        tours = torch.stack(list, dim=0)
        return tours

    def stack_l(self, inputs, tours):
        '''
        inputs:(batch,city_t,2)
        tours:(batch,city_t)
        return l_batch:(batch)
        '''
        list = [self.get_tour_distance(inputs[i], tours[i]) for i in
                range(self.batch)]
        l_batch = torch.stack(list, dim=0)
        return l_batch

    def stack_l_fast(self, inputs, tours):
        """
        *** this function is faster version of stack_l! ***
        inputs: (batch, city_t, 2), Coordinates of nodes
        tours: (batch, city_t), predicted tour
        d: (batch, city_t, 2)
        """
        d = torch.gather(input=inputs, dim=1,
                         index=tours[:, :, None].repeat(1, 1, 2))
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - d[:, -1]).norm(p=2,
                                            dim=1))  # distance from last node to first selected node)

    def seq_score(self, inputs, tours):  # fixme: 排序只需要做一次就行，内部使用顺序直接切片加上堆的做法
        """ # todo: tensor 数据拉回 cpu ，转换为numpy，算完后扔回gpu，多线程运算 
        inputs: (task_data, env_data), tuple of task_data and env_data
        tours: (batch, city_t), predicted orders
        d: (batch) (batch, slots, tasks) (batch, slots, tasks) -> batch * score
        """  # todo: 迁移学习
        INF = 44444444  # fixme: 整体模块的可使用性不能完全确定

        batch, _, tasks_slots_size = inputs.size()
        task_data, env_data = torch.split(inputs, dim=2,
                                          split_size_or_sections=[
                                              tasks_slots_size - self.slots * 2,
                                              self.slots * 2])
        # 这里的 indices_or_sections 是不是应该要变为数据的实际长度
        env_data = env_data[:, 0, :]
        task_data = task_data.cpu().numpy()
        env_data = env_data.cpu().numpy()
        tours = tours.cpu().numpy()  # 先将输入转为numpy类型
        batch, c_p_size = env_data.shape  # torch 的size 对应 numpy 的shape
        slots_size = c_p_size // 2  # 时隙数量 #fixme: 保证inputs，tours的batch_size一致性
        _, task_size = tours.shape

        c_slots, p_slots = np.split(env_data, 2, axis=1)
        c_sum_slots = np.cumsum(c_slots, axis=1)  # 对最后一个维度进行累加

        # c_slots, c_sum_slots, env_data, p_slots 的有效数据从0开始使用
        dp = np.full((batch, task_size, slots_size), fill_value=-INF,
                     dtype=np.float32)
        real_worker = np.full((batch, slots_size, slots_size), fill_value=0,
                              dtype=np.int)  # (batch, slots_size, slots_size) i： 多少批次 j：以 j 为 last_t 方案 k：具体方案的具体实施方法
        total_utility = np.full((batch, task_size, slots_size), fill_value=0,
                                dtype=np.float32)  # 和 dp 模块等大小,记录整体的 utility
        # worker_trace = np.full((batch, task_size, slots_size), fill_value=0, dtype=np.int) # 完整记录每个时间点上每个任务分配了多少资源

        workload, penalty_factor = np.split(task_data, 2,
                                            axis=2)  # workload then penalty_factor
        workload = np.squeeze(workload, axis=-1)  # 降解最后一个维度
        penalty_factor = np.squeeze(penalty_factor, axis=-1)
        for b in range(batch):
            remain_cpu = [0 for _ in range(slots_size)]
            for i in range(task_size):
                cur_task_index = tours[b][i]
                if i == 0:  # 特殊处理第一个任务
                    j = slots_size - 1
                    while j >= 0:  # 从后往前遍历
                        if (c_sum_slots[b][j] >= workload[b][cur_task_index]):
                            worker = np.full((slots_size,), fill_value=0,
                                             dtype=np.int)  # 记录可行的一个方案, 针对 i 任务的可行方案
                            cost = 0
                            utility = 0
                            last_t = 0
                            rank = p_slots[b][
                                   :j + 1].argsort()  # 按照rank从左到右使用时隙
                            rank = iter(rank)
                            workload_temp = workload[b][cur_task_index]
                            p_min_slot = -1
                            while (workload_temp > 0):
                                p_min_slot = next(rank, -1)
                                last_t = max(last_t, p_min_slot)
                                workload_temp = workload_temp - c_slots[b][
                                    p_min_slot]
                                cost = cost + c_slots[b][p_min_slot] * \
                                       p_slots[b][p_min_slot]
                                worker[p_min_slot] = c_slots[b][
                                    p_min_slot]  # 某一个方案中使用了某一个时隙的资源，有多少
                            utility = -last_t * penalty_factor[b][
                                cur_task_index]
                            if p_min_slot == last_t and workload_temp < 0:  # 最后使用的时隙是 last_t 并且有资源的剩余
                                worker[last_t] = worker[
                                                     last_t] + workload_temp  # 第一个任务的 last_t
                                cost = cost + workload_temp * p_slots[b][last_t]
                                workload_temp = 0
                                if utility - cost > dp[b][0][
                                    last_t]:  # 最后的时隙无剩余完成任务
                                    real_worker[b][last_t] = copy.deepcopy(
                                        worker)  # 刚开始完全复制 是没有问题的, 而且一定不需要考虑 last_t 重复的问题
                                    dp[b][0][last_t] = utility - cost
                                    total_utility[b][0][
                                        last_t] = utility  # 第一个任务, 先记录所有的 utility
                                    remain_cpu[last_t] = -workload_temp
                            if p_min_slot == -1:  # 初始任务负载为0: 如果进入这里 默认的是 last_t 就是0
                                remain_cpu[last_t] = c_slots[b][
                                    last_t]  # 如果说使用的成本很高呢？一定要牢记使用量必定是 0
                                worker[last_t] = 0  # 同样的可行方案中 有一个的使用量是 0
                                total_utility[b][0][last_t] = 0
                                real_worker[b][last_t] = copy.deepcopy(worker)
                                workload_temp = 0
                            # dp[b][0][last_t] = max(dp[b][0][last_t], (utility - cost)) # total u - c
                            # 如果选择的是 u - c 则无理由的替换掉 real_worker[b][last]
                            if p_min_slot != -1:  # 当前任务的工作量大于 0
                                cost = cost + workload_temp * p_slots[b][
                                    p_min_slot]  # 有一部分的 c_slots[b][p_min_slot] 是没有用完的
                            if utility - cost > dp[b][0][
                                last_t]:  # 完成任务的最后一个时隙不是 last_t 是在中间的某一个位置
                                if p_min_slot != -1:
                                    worker[p_min_slot] = worker[
                                                             p_min_slot] + workload_temp
                                real_worker[b][last_t] = copy.deepcopy(
                                    worker)  # 刚开始完全复制 是没有问题的, 而且一定不需要考虑 last_t 重复的问题
                                dp[b][0][last_t] = utility - cost
                                total_utility[b][0][
                                    last_t] = utility  # 第一个任务, 先记录所有的 utility
                                remain_cpu[last_t] = 0
                            # last_t 包括以后的时隙全部都不需要管了
                            j = last_t - 1
                        else:
                            j -= 1

                    largest_num_left = -INF  # 记录左侧的最大值
                    slots_posible_stratage = np.array([],
                                                      dtype=np.int)  # 记录可行方案的时刻
                    for j in range(slots_size):  # 清除右边比左边小的值
                        if dp[b][0][j] <= largest_num_left:
                            dp[b][0][j] = -INF
                            total_utility[b][0][j] = 0  # 不具备可行方案
                            real_worker[b][j] = [0 for _ in range(
                                slots_size)]  # 如果说没有可行方案, 将last_t对应的方案全部置零
                        else:
                            largest_num_left = dp[b][0][j]
                            slots_posible_stratage = np.append(
                                slots_posible_stratage,
                                j)  # 对应的位置才需要查询是否有remain的cpu资源, 下面的对应方法一定是将第一层的worker 做过清除
                else:  # i > 0
                    largest_num_left = -INF  # 记录左侧的最大值
                    remain_temp = [0 for _ in range(
                        slots_size)]  # 从此处开始使用 remain_temp，作用是记录完成这个任务过程中有哪些时隙是最后使用的并且有可以使用的剩余资源
                    worker_temp = np.full((batch, slots_size, slots_size),
                                          fill_value=0,
                                          dtype=np.int)  # 记录本轮任务完成时的完整使用轨迹
                    for j in slots_posible_stratage:  # 可能的左端点
                        right = slots_size
                        while (
                                right > j):  # 每进入一次， remain_cpu 都应该和上一个任务结束时的完全一致
                            cost = 0
                            utility = 0
                            if remain_cpu[
                                j] > 0:  # 上面的所有任务在j完成，并且在j时隙还有剩余的资源可以使用
                                if (c_sum_slots[b][right - 1] - c_sum_slots[b][
                                    j] + remain_cpu[j]) >= workload[b][
                                    cur_task_index]:
                                    worker = np.full((slots_size,),
                                                     fill_value=0, dtype=np.int)
                                    last_t = 0
                                    rank = p_slots[b][j + 1: right].argsort()
                                    rank = rank + j + 1  # 可以从 j 时隙开始使用
                                    rank = iter(rank)
                                    workload_temp = workload[b][cur_task_index]
                                    p_min_slot = -1
                                    remain_cpu_term = copy.deepcopy(
                                        remain_cpu)  # remain_cpu_term上一个任务结束后有多少可以使用的资源，可以在不同的起始端点重复使用remain_cpu_term，对它进行修改相当于一个标志，用来辅助修改cpu_temp
                                    if workload_temp <= 0:  # 跳过workload为0的情况
                                        if dp[b][i - 1][j] > dp[b][i][
                                            j]:  # 涉及对 remain_temp[j] 的重复赋值，只有当 dp 的取值更加时，对 remain_temp[j] 重复赋值
                                            dp[b][i][j] = dp[b][i - 1][j]
                                            total_utility[b][i][j] = \
                                                total_utility[b][i - 1][j] - j * \
                                                penalty_factor[b][
                                                    cur_task_index]  # 整体的utility会增加一个在  j 时刻结束的 分量
                                            remain_temp[j] = remain_cpu[
                                                j]  # 最后一个时隙完成了任务，并且还有剩余的资源
                                            worker_temp[b][j] = real_worker[b][
                                                j]  # 对最后一个时隙的重复
                                        # worker = copy.deepcopy(real_worker[b][j]) # 即使仍然需要赋值，需要做的事情是 real_worker[b][j] = real_worker[b][j]
                                    else:  # 如果说 workload 非 0
                                        workload_temp -= remain_cpu_term[
                                            j]  # 首先使用一部分 cpu 资源, 但是在while循环中会重复使用 remain_cpu 的资源, 因此在排序的时候应该和remain_cpu为0的方式相同考虑
                                        cost = cost + remain_cpu_term[j] * \
                                               p_slots[b][
                                                   j]  # 首先得默认 remain_cpu 使用完毕
                                        worker[j] += remain_cpu_term[
                                            j]  # 默认使用 j 时隙的所有剩余 cpu 资源, 首先使用了 remain 的资源，但是问题是以62结尾的应该没有剩余的资源了，为什么还要继续使用呢
                                        if workload_temp < 0:  # 依旧无法使用完 remain_cpu[j] ，j 依旧是最后的一个时隙
                                            remain_cpu_term[
                                                j] = -workload_temp  # 给本轮使用
                                            worker[
                                                j] += workload_temp  # 剩余 cpu 资源多的退回去
                                            last_t = j
                                            utility = -last_t * \
                                                      penalty_factor[b][
                                                          cur_task_index]
                                            cost = cost + workload_temp * \
                                                   p_slots[b][j]
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) >= dp[b][i][
                                                last_t]:  # 新的分配方案更优
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                                remain_temp[
                                                    j] = -workload_temp  # remain_temp[j] 是否会重复赋值，应该选取其中的哪一个？应该选取dp数值更大的那一个
                                                worker_temp[b][last_t] = \
                                                    real_worker[b][
                                                        j] + worker  # 上次结束使用的时隙是 j , 现在还是在使用 last_t的时隙
                                            workload_temp = 0  # 资源反还完毕，刚好完成任务
                                        while (
                                                workload_temp > 0):  # 使用完了上个任务剩余时隙的资源,接着还需要找
                                            p_min_slot = next(rank, -1)
                                            last_t = max(last_t, p_min_slot)
                                            workload_temp = workload_temp - \
                                                            c_slots[b][
                                                                p_min_slot]
                                            cost = cost + c_slots[b][
                                                p_min_slot] * p_slots[b][
                                                       p_min_slot]
                                            worker[p_min_slot] = c_slots[b][
                                                p_min_slot]  # 记录在某一个时隙使用了多少资源
                                        utility = -last_t * penalty_factor[b][
                                            cur_task_index]
                                        if p_min_slot == last_t and workload_temp < 0:  # 任务完成了，并且最后的一个时隙有剩余cpu的情况
                                            worker[last_t] = worker[
                                                                 last_t] + workload_temp  # 最后一个时隙多使用的资源不能记录在worker
                                            cost = cost + workload_temp * \
                                                   p_slots[b][
                                                       last_t]  # 发现有一部分资源没有使用完
                                            workload_temp = 0  # 完成该任务
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) > dp[b][i][
                                                last_t]:  # 只在这种情况下进行重新赋值是否正确呢？ 我的答案应该是正确的
                                                worker_temp[b][
                                                    last_t] = worker + \
                                                              real_worker[b][
                                                                  j]  # 上一次的 j 还有残留 不用考虑和下面的 if 重叠的状态
                                                remain_temp[
                                                    last_t] = -workload_temp  # 这种情况下需要对remain_temp 进行更新
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                        if p_min_slot != -1:  # 当前任务的工作量大于 0
                                            cost = cost + workload_temp * \
                                                   p_slots[b][p_min_slot]
                                        if ((utility - cost) + dp[b][i - 1][
                                            j]) > dp[b][i][
                                            last_t]:  # 这种状态下说明最后完成的 slot 没有剩余的资源，仅仅是对 dp 进行一次更新
                                            if p_min_slot != -1:
                                                worker[
                                                    p_min_slot] += workload_temp  # 多使用了一部分的 c_slots 资源
                                            worker_temp[b][last_t] = worker + \
                                                                     real_worker[
                                                                         b][
                                                                         j]  # 上一次的 j 还有残留
                                            dp[b][i][last_t] = (
                                                    (utility - cost) +
                                                    dp[b][i - 1][j])
                                            total_utility[b][i][last_t] = \
                                                total_utility[b][i - 1][
                                                    j] + utility
                                            remain_temp[last_t] = 0
                                    right = last_t
                                else:
                                    right -= 1
                            else:  # remain_cpu = 0
                                if (c_sum_slots[b][right - 1] - c_sum_slots[b][
                                    j]) >= workload[b][
                                    cur_task_index]:  # 从时隙 j + 1开始能满足工作负载要求
                                    worker = np.full((slots_size,),
                                                     fill_value=0, dtype=np.int)
                                    last_t = 0
                                    rank = p_slots[b][j + 1: right].argsort()
                                    rank = rank + j + 1
                                    rank = iter(rank)
                                    workload_temp = workload[b][cur_task_index]
                                    remain_cpu_term = copy.deepcopy(remain_cpu)
                                    if workload_temp <= 0:  # 当前的任务需要的资源量是 0 , 最后一个完成的时隙是 j, 且完成了任务
                                        if dp[b][i - 1][j] > dp[b][i][j]:
                                            dp[b][i][j] = dp[b][i - 1][j]
                                            total_utility[b][i][j] = \
                                                total_utility[b][i - 1][j] - j * \
                                                penalty_factor[b][
                                                    cur_task_index]
                                            remain_temp[j] = remain_cpu[j]
                                            worker_temp[b][j] = real_worker[b][
                                                j]
                                    else:  # 当前任务的资源需求量大于0
                                        p_min_slot = -1
                                        while (workload_temp > 0):
                                            p_min_slot = next(rank, -1)
                                            last_t = max(last_t, p_min_slot)
                                            workload_temp = workload_temp - \
                                                            c_slots[b][
                                                                p_min_slot]
                                            worker[p_min_slot] = c_slots[b][
                                                p_min_slot]  # 记录下某一个时隙使用了具体多少资源
                                            cost = cost + c_slots[b][
                                                p_min_slot] * p_slots[b][
                                                       p_min_slot]
                                        utility = -last_t * penalty_factor[b][
                                            cur_task_index]
                                        if p_min_slot == last_t and workload_temp < 0:  # 在最后的一个时隙完成了任务，并且需要更新 remain_temp 和 dp
                                            worker[last_t] = worker[
                                                                 last_t] + workload_temp  # 实际环境少使用的资源需要吐出来
                                            cost = cost + workload_temp * \
                                                   p_slots[b][
                                                       p_min_slot]  # 少使用的花费
                                            workload_temp = 0
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) >= dp[b][i][last_t]:
                                                remain_temp[
                                                    last_t] = -workload_temp
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                                worker_temp[b][last_t] = (
                                                        worker +
                                                        real_worker[b][j])
                                        if p_min_slot != -1:  # 当前任务的工作量大于 0
                                            cost = cost + workload_temp * \
                                                   p_slots[b][p_min_slot]
                                        if ((utility - cost) + dp[b][i - 1][
                                            j]) >= dp[b][i][
                                            last_t]:  # 做的事情完成任务的时隙不是最后的时隙，在这种情况下，只需要更新 dp, remain_cpu 不需要更新, workload_temp 同样可能是小于 0 的
                                            if p_min_slot != -1:
                                                worker[
                                                    p_min_slot] += workload_temp
                                            dp[b][i][last_t] = (
                                                    (utility - cost) +
                                                    dp[b][i - 1][
                                                        j])  # 上一个任务在 j 时隙已经完成了
                                            total_utility[b][i][last_t] = \
                                                total_utility[b][i - 1][
                                                    j] + utility
                                            worker_temp[b][last_t] = (
                                                    worker + real_worker[b][
                                                j])  # 全新的数, j 是上一个任务完成的时隙
                                            remain_temp[
                                                last_t] = 0  # 需要清空最后一个时隙的资源
                                    right = last_t
                                else:
                                    right -= 1
                    slots_posible_stratage = np.array([],
                                                      dtype=np.int)  # 记录可行方案的时刻
                    for j in range(slots_size):  # 清除右边比左边小的值
                        if dp[b][i][j] <= largest_num_left:
                            dp[b][i][j] = -INF
                            total_utility[b][i][j] = 0
                            worker_temp[b][j] = [0 for _ in range(slots_size)]
                            remain_temp[j] = 0
                        else:
                            largest_num_left = dp[b][i][j]
                            slots_posible_stratage = np.append(
                                slots_posible_stratage, j)
                    remain_cpu = copy.deepcopy(remain_temp)  # 此处更新位置没有错
                    real_worker = copy.deepcopy(
                        worker_temp)  # 此处还需要更新 real_worker 的数据
        d = np.full((batch,), fill_value=-INF, dtype=np.float32)
        u = np.full((batch,), fill_value=0, dtype=np.float32)
        actor_source = np.full((batch, slots_size), fill_value=0, dtype=np.int)
        for b in range(batch):
            d[b] = -np.max(dp[b][task_size - 1])
            max_index = np.argmax(
                dp[b][task_size - 1])  # 无可行方案时如何记录, 无可行方案时, d[b] 返回的是
            actor_source[b] = copy.deepcopy(
                real_worker[b][max_index])  # 如果出现没有可行方案怎么办？是否应该返回无法执行
            u[b] = total_utility[b][task_size - 1][max_index]
            if (-np.max(dp[b][task_size - 1]) == INF):
                d[b] = 5000
                u[b] = 0  # 如果没有可行方案, 则返回的utility为0
        trace = np.full((batch, task_size, slots_size), fill_value=0,
                        dtype=np.int)
        for b in range(batch):
            slot = 0  # 从 0 时隙 遍历到第 99 时隙
            slot_over_flag = False
            for i in range(len(tours[b])):  # 第 i 个任务
                cur_task_index = tours[b][i]
                workload_temp = workload[b][cur_task_index]
                while workload_temp > 0:  # 当前任务没有完成
                    if actor_source[b][
                        slot] > 0:  # 有时隙资源 如果说没有可行方案，还需要管 trace 的计算吗？如何判断没有可行方案？ 根据打分大小吗？
                        trace[b][i][slot] = actor_source[b][slot]
                        workload_temp -= actor_source[b][slot]
                    slot += 1  # 每判断一次递进到下一个 slot
                    if slot == slots_size:
                        slot_over_flag = True
                        break
                if slot_over_flag:
                    break

        # print('seq_score down')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.from_numpy(d).to(device), u, trace  # 一定要返回cuda数据（gpu)

    def show(self, nodes, tour):
        nodes = nodes.cpu().detach()
        print('distance:{:.3f}'.format(self.get_tour_distance(nodes, tour)))
        print(tour)
        plt.figure()
        plt.plot(nodes[:, 0], nodes[:, 1], 'yo', markersize=16)
        np_tour = tour[:].cpu().detach()
        np_fin_tour = [tour[-1].item(), tour[0].item()]
        plt.plot(nodes[np_tour, 0], nodes[np_tour, 1], 'k-', linewidth=0.7)
        plt.plot(nodes[np_fin_tour, 0], nodes[np_fin_tour, 1], 'k-',
                 linewidth=0.7)
        for i in range(self.city_t):
            plt.text(nodes[i, 0], nodes[i, 1], str(i), size=10, color='b')
        plt.show()

    def shuffle(self, inputs):
        '''
        shuffle nodes order with a set of xy coordinate
        inputs:(batch,city_t,2)
        return shuffle_inputs:(batch,city_t,2)
        '''
        batch, city_t, _ = inputs.size()
        shuffle_inputs = torch.zeros(inputs.size())
        for i in range(batch):
            perm = torch.randperm(self.city_t)
            shuffle_inputs[i, :, :] = inputs[i, perm, :]
        return shuffle_inputs

    def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs):
        '''
        pred_shuffle_tours:(batch,city_t)
        shuffle_inputs:(batch,city_t_t,2)
        test_inputs:(batch,city_t,2)
        return pred_tours:(batch,city_t)
        '''
        batch, _ = pred_shuffle_tours.size()
        pred_tours = []
        for i in range(batch):
            pred_tour = []
            for j in range(self.city_t):
                xy_temp = shuffle_inputs[0][i][pred_shuffle_tours[i][j]]
                for k in range(self.city_t):
                    if torch.all(torch.eq(xy_temp, test_inputs[i][k])):
                        pred_tour.append(torch.tensor(k))
                        if len(pred_tour) == self.city_t:
                            pred_tours.append(torch.stack(pred_tour, dim=0))
                        break
        pred_tours = torch.stack(pred_tours, dim=0)
        return pred_tours

    def get_tour_distance(self, nodes, tour):
        '''
        nodes:(city_t,2), tour:(city_t)
        l(= total distance) = l(0-1) + l(1-2) + l(2-3) + ... + l(18-19) + l(19-0) @20%20->0
        return l:(1)
        '''
        l = 0
        for i in range(self.city_t):
            l += get_2city_distance(nodes[tour[i]],
                                    nodes[tour[(i + 1) % self.city_t]])
        return l

    def get_random_tour(self):  # fixme:默认城市是从0开始的随机产生路途,和我们的随机不同，返回任务编号，效果一致
        '''
        return tour:(city_t)
        '''
        tour = []
        # 这里的 city_t 可能是改变的那么神经网络需要改变吗? 实际上是不需要的
        while set(tour) != set(range(self.city_t)):  # 这里的city_t需要根据实际改变了
            city = np.random.randint(self.city_t)
            if city not in tour:
                tour.append(city)
        tour = torch.from_numpy(np.array(tour))
        return tour

    def get_optimal_tour(self, nodes):
        # dynamic programming algorithm, calculate lengths between all nodes
        points = nodes.numpy()
        all_distances = [[get_2city_distance(x, y) for y in points] for x in
                         points]
        # initial value - just distance from 0 to every other point + keep the track of edges
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for
             idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in
                      itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j],
                                      A[(S - {j}, k)][1] + [j]) for k in S if
                                     k != 0 and k != j])  # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
            A = B
        res = min(
            [(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        tour = torch.from_numpy(np.array(res[1]))
        return tour


def random_train_task(batch, city_t, seed=None):
    '''
    任务随机训练数据
    return nodes:(batch,city_t,2)
    '''
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task_workload = 10 * torch.rand((batch, city_t, 1), device=device) + 5
    task_penalty_factor = torch.rand((batch, city_t, 1),
                                     device=device)  # fixme: task 参数名需要改变
    task_info = torch.cat((task_workload, task_penalty_factor), 2)
    return task_info  # 可以模仿的产生 通过 cat 拼接 两次产生的tensor


def random_train_env(batch, slots, seed=None):
    '''
    环境随机训练数据
    return node:(batch,slots * 2)
    '''
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env_computation_power = 10 * torch.rand((batch, slots, 1),
                                            device=device) + 5
    env_price = 9 * torch.rand((batch, slots, 1),
                               device=device) + 1  # fixme: env 参数名需要改变
    env_info = torch.cat((env_computation_power, env_price), 2)
    # reshape env to (batch, slots * 2)
    env_info = torch.reshape(env_info, (batch, slots * 2))
    return env_info  # 可以模仿的产生 通过 cat 拼接 两次产生的tensor


if __name__ == '__main__':
    random_train_task(batch=64, city_t=5)
    random_train_env(batch=64, slots=10)
    print('hello world')
