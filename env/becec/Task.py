import numpy as np
class Task:
    """
        A class used to describe the task

        Attributes:
            arrival_time:               Task arrival time
            w = [5M, 20M]:             The clock cycle required by the task
            u_0 = [100, 1000]:            top of utility
            alpha = [5, 25]:         coefficient of utility
    """
    def __init__(self, arrival_time: int, w=1, u_0=10., alpha=1., random_generate=False):
        self.arrival_time = arrival_time
        
        if random_generate:
            self.random_generate()
        else:
            self.w = w          # 别直接取 w
            self.u_0 = u_0
            self.alpha = alpha

    def random_generate(self):
        self.w = np.random.randint(5, 21) # [5, 20]
        # 保留三位小数
        # self.u_0 = np.random.randint(100000, 500001) / 1000.      # [100, 500]
        self.u_0 = np.random.randint(0, 500001) / 1000. # [0, 500]
        self.alpha = np.random.randint(5000, 25001) / 1000.      # [5, 25]

    def utility(self, t: int):
        """
        计算任务的效益
        :param t:   任务完成时间
        :return:
        """
        if t < self.arrival_time:
            print(f"The task completion time t={t} was passed in incorrectly! (arrival_time={self.arrival_time})")
            return 0
        return self.u_0 - self.alpha * (t - self.arrival_time)

    def cpu_requirement(self):
        # 1 MCycles of Task = 1 GHz of CPU
        # We should use 1 MHz as a unit of cpu (multiple 1e3)
        return self.w * 1e3
