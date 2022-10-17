import numpy as np
from .Task import Task
from .MMPP import MMPP

MMPP_MEAN_RESIZE = 0.95   # mmpp 均值的调整系数，越小任务生成率越大

C_INTERNAL_FLOOR = 20 * 1e3  # MHz
C_INTERNAL_CEIL = 40 * 1e3  # MHz

C_INITIAL_RATE = 0.5        # 初始负载在总负载的 [C_INITIAL_RATE, 1.]

class BaseStation:
    """
        A class used to describe a Base Station

        Attributes:
            timer:                                  record present time slot number
            delta_t:                                number of future slots the algorithm use to allocate at once
            tasks_internal:                         tasks generated by this BS in the next delta_t slots
            tasks_external:                         tasks scheduled to this BS in this timer
            p_coef:                                 unit resource price for the state of empty occupation
            C_internal:                             max CPU capacity
            load_internal:                          CPU resources used for self at a given slot
            load_external:                          CPU resources used to process tasks_external at a given slot
    """

    def __init__(self, BS_ID: int, config):
        """
        :param delta_t: 用于调度的时间区间长度
        """
        self.uuid = BS_ID
        self.config = config
        self.tasks_internal = [[] for _ in range(config['delta_t'])]  # 记录从当前 timer 开始的 delta_t 个 slot 下的本 BS 任务
        self.tasks_external = []  # 记录当前 timer 的到达任务（stage 1 可能给该 BS 分配多个任务，在此缓存，以便 stage 2 进行调度）
        # 进队列的任务还未在 load_external 中被记录，因此新任务的加入需要考虑这个队列！！！
        
        self.timer = 0
        self.helper_timer = 0

        self.p_coef = np.random.randint(50, 100) / 100. / 1e3  # p_coef[i] = [0.5, 1] ($/GHz)

        self.C_internal = np.random.randint(C_INTERNAL_FLOOR, C_INTERNAL_CEIL + 1)
        
        # Markov Modulated Poisson Processes
        self.mmpp = MMPP(max_load_internal=self.C_internal)
        
        self.load_external = [0 for _ in range(config['delta_t'])]

        # 初始化 delta_t 个 slot 的内部 load
        self.load_internal = [0 for _ in range(config['delta_t'])]
        self.load_internal[0] = np.random.randint(int(C_INITIAL_RATE * self.C_internal), self.C_internal + 1)  # 初始负载
        for j in range(1, config['delta_t']):
            self.load_internal[j], self.tasks_internal[j] = self._get_new_load_interval(t=j)
    
    def reset(self):
        # 仅重置各类缓存，不重新生成基站的属性（比如 mmpp、p_coef）
        delta_t = self.config['delta_t']
        self.timer = 0
        self.helper_timer = 0
        self.tasks_internal = [[] for _ in range(delta_t)]
        self.tasks_external = []
        self.load_external = [0 for _ in range(self.config['delta_t'])]
        self.load_internal = [0 for _ in range(self.config['delta_t'])]
        self.load_internal[0] = np.random.randint(int(C_INITIAL_RATE * self.C_internal), self.C_internal + 1)  # 初始负载
        for j in range(1, delta_t):
            self.load_internal[j], self.tasks_internal[j] = self._get_new_load_interval(t=j)

    def _generate_task(self, arrival_time: int):
        """
        生成新任务
        :param arrival_time:    任务的到达时间
        :return:                返回新任务对象
        """
        task = Task(arrival_time=arrival_time, random_generate=True)
        return task

    def _get_new_load_interval(self, t: int):
        """
        根据上一个 slot 的内部负载获取新的内部负载
        :param t:                         time slot in the future, should be in the range of [0, delta_t - 1], which is an offset from the present timer
        :return:                          [新的内部负载, 新生成的任务队列]
        """
        
        def uniform_arrival():
            # 以 1/3 的概率增加/减少/不变
            rand_int = np.random.randint(0, 3)
            if rand_int == 0:
                difference = int(0.1 * self.C_internal)
            elif rand_int == 1:
                difference = -int(0.1 * self.C_internal)
            else:
                difference = 0
            return difference
        
        def mmpp_arrival():
            # 为每个BS维护一个arrival process，并且假设每个BS的处理速度和平均到达速率相等，difference[i][t] = arrival[i][t] - process_rate[i]，其中process_rate[i] = E(arrival[i][t])。
            self.mmpp.next_state()
            num_of_arrivals = self.mmpp.generate_arrivals()
            mean_of_arrivals = self.mmpp.mean_arrival
            difference = num_of_arrivals - mean_of_arrivals*MMPP_MEAN_RESIZE # let arrivals rate greater than consuming speed
            return difference
        
        def simple_arrival():
            # 每个 slot 的总负载在 [0.5, 1.25] 之间均匀取值
            new_rate = np.random.randint(50, 126) / 100.
            new_load = self.C_internal * new_rate
            old_load = self.load_internal[t-1] + self.load_external[t]
            difference = new_load - old_load
            return difference
        
        old_t = t - 1
        arrival_time = self.timer + t
        # difference = uniform_arrival()
        # difference = mmpp_arrival()
        difference = simple_arrival()
        new_load_interval = max(0, self.load_internal[old_t] + difference)

        total_load = new_load_interval + self.load_external[t]
        tasks = []
        if total_load > self.C_internal:
            # 超出负荷，生成新任务
            # 如果负载超出capacity，多出的那一部分会生成若干任务提交给其他BS。
            # （弃用）假如我们超的负载是3G，对应的循环数是3M（每个slot时长1ms），由于每个任务平均的cycles是30M，所以我们会生成0.1个任务（以0.1的概率生成一个task，这个task的workload仍然服从均匀分布，而不是固定30M）
            # （弃用）假如我们超的负载是4.5G，对应的循环数是4.5M（每个slot时长1ms），由于每个任务平均的cycles是3M，所以我们会生成1.5个任务（先生成一个任务，然后以0.5的概率生成另一个task，这个task的workload仍然服从均匀分布，而不是固定3M）
            overload = total_load - self.C_internal
            threshold = 1e-3 * overload / 12.5  # 除的是任务 w 的均值
            if threshold > 1:
                # print(f"timer {self.timer} has more than one task in a BS!")
                while threshold > 1:
                    threshold -= 1
                    tasks.append(self._generate_task(arrival_time=arrival_time))
            rand_num = np.random.randint(1, 101) / 100.
            if rand_num <= threshold:
                tasks.append(self._generate_task(arrival_time=arrival_time))

            new_load_interval = self.C_internal - self.load_external[t]

        return [new_load_interval, tasks]

    def base_unit_price(self):
        """
        获取单位资源价格的基准价格（资源空闲时的单价）
        :return:
        """
        return self.p_coef

    def unit_price(self, t: int):
        """
        返回从当前时刻开始，之后第 t 个 slot 的单位资源价格
        :param t:                         time slot in the future, should be in the range of [0, delta_t - 1], which is an offset from the present timer
        :return:
        """
        if self.cpu_remain(t=t) > 0:
            # p[i][t] = 1 / {1 - (load_internal[i][t] + load_external[i][t]) / C_internal[i]} * p_coef[i]
            return self.base_unit_price() / (1. - (self.cpu_load(t=t) / self.cpu_capacity()))
        else:
            # 没有剩余资源，则返回 1，表示资源不出售
            return 1.

    def cpu_capacity(self):
        """
        获取 CPU 的资源上限
        :return:
        """
        return self.C_internal

    def cpu_load(self, t: int):
        """
        返回从当前时刻开始，之后第 t 个 slot 的 CPU 资源占用量
        :param t:                         time slot in the future, should be in the range of [0, delta_t - 1], which is an offset from the present timer
        :return:
        """
        return self.load_internal[t] + self.load_external[t]

    def cpu_remain(self, t: int):
        """
        返回从当前时刻开始，之后第 t 个 slot 的 cpu 空闲资源值
        如果负载小于capacity的80%，那么我们就会把多余的那一部分，也就是0.8*C_internal[i] - load_internal[i][t] - load_external[i][t]的部分，作为空闲资源报告给controller
        :param t:                         time slot in the future, should be in the range of [0, delta_t - 1], which is an offset from the present timer
        :return:
        """
        if not 0 <= t < self.config['delta_t']:
            print(f"Parameter slot={t} is out of range!")
            return -1.

        if self.cpu_load(t) <= 0.8 * self.cpu_capacity():
            return int(0.8 * self.cpu_capacity() - self.cpu_load(t))
        else:
            return 0.

    def next(self):
        """
        进入下一个 slot，更新 timer、清理 cache、更新负载情况
        :return:                        A list of tasks generates by busy BSs in the next timer
        """
        self.timer += 1
        if len(self.tasks_external):
            self.clear_tasks()

        # 过了一个时隙，弹出资源队列中的过时元素，并压入一个新的
        self.load_internal.pop(0)
        self.load_internal.append(0)

        self.load_external.pop(0)
        self.load_external.append(0)

        self.tasks_internal.pop(0)
        self.tasks_internal.append([])

        # 计算刚压入的新元素（第 delta_t 个 slot）
        self.load_internal[-1], self.tasks_internal[-1] = self._get_new_load_interval(t=self.config['delta_t'] - 1)

        if self.cpu_remain(0) > 0.:
            self.helper_timer += 1

        return self.tasks_internal[0]

    def is_resource_enough(self, task: Task):
        """该方法用于第一阶段调度
        判断新来的需求加上缓存区的任务，该 BS 的资源是否够用
        :param task:
        :return:                            True / False
        """
        if task is None:
            print(f"Parameter is invalid!")
            return False

        # 新任务的加入需要考虑已经加入队列的任务，因为 cpu_remain 只考虑已经被分配的任务，此时处于第一阶段，任务都只是被缓存
        require_total = sum(t_e.cpu_requirement() for t_e in self.tasks_external) + task.cpu_requirement()
        resource_remain = sum(self.cpu_remain(t=t) for t in range(self.config['delta_t']))

        return require_total <= resource_remain

    def is_resource_enough_at_slots(self, task: Task, slots_begin: int, slots_end: int):
        """该方法用于第二阶段分配
        判断当前 BS 在未来给定的时隙中存在满足要求的资源量
        :param task:
        :param slots_begin:              the first slot number [0, delta_t - 1]
        :param slots_end:                the last slot number [0, delta_t - 1]
        :return:                            True / False
        """
        if not (
                0 <= slots_begin < self.config['delta_t'] and 0 <= slots_end < self.config['delta_t'] and slots_begin <= slots_end) or task is None:
            print(f"Parameter is invalid!")
            return False

        # 因为第二阶段已经开始进行任务分配，cpu_requirement 里考虑了已经被分配的任务信息
        require_total = task.cpu_requirement()
        resource_remain = sum(self.cpu_remain(t=t) for t in range(slots_begin, slots_end + 1))

        return require_total <= resource_remain

    def schedule_task(self, task: Task):
        """该方法用于第一阶段调度
        将任务调度给当前基站
        :param task:
        :return:                        0 for success, -1 for failure
        """
        ret = 0
        # if not self.is_resource_enough(task=task):
        #     # 第一阶段的分配出现了问题，如果要使用 CMDP，可能需要在这里进行一些操作
        #     ret = -1
        self.tasks_external.append(task)        # 在第二阶段对无法分配的任务进行惩罚，因此始终添加任务
        return ret

    def price_of_allocation(self, alloc_list):
        """
        判断一个调度策略的可行性，并返回其价格开销（不会执行调度）
        :param alloc_list:              the allocation for task in this BS during the next 0 to delta_t - 1 slots. len = delta_t
        :return:                        total price of this allocation, -1 for failure
        """
        if len(alloc_list) != self.config['delta_t']:
            print(f"Parameter is invalid!")
            return -1

        price = 0
        # 验证每个 slot 的调度是否合法
        for t in range(self.config['delta_t']):
            alloc = alloc_list[t]
            if self.cpu_remain(t=t) < alloc:
                print(f"Allocation resource quantity error! Remained resource is less than required.")
                return -1
            price += alloc * self.unit_price(t=t)

        return price

    def allocate_task(self, task: Task, alloc_list):
        """该方法用于第二阶段分配
        将任务按照调度列表分配给当前基站
        :param task:
        :param alloc_list:              the allocation for task in this BS during the next 0 to delta_t - 1 slots. len = delta_t
        :return:                        total price of this allocation, -1 for failure
        """
        price = self.price_of_allocation(alloc_list=alloc_list)

        if price != -1:
            # 执行调度
            for t in range(self.config['delta_t']):
                self.load_external[t] += alloc_list[t]
            # 移除任务
            self.tasks_external.remove(task)

        return price

    def clear_tasks(self):
        """当任务无法在第二阶段被分配时，执行此方法清除任务
        """
        # print(f"There are {len(self.tasks_external)} external tasks not been allocated!")
        self.tasks_external.clear()

    def seed(self, seed):
        np.random.seed(seed)