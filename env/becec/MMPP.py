import numpy as np

class LinearSolver:
    @classmethod
    def AX_equal_b(self, A, b):
        """求解非齐次方程 AX = b
        Args:
            A (np.ndarray): 系数矩阵
            b (np.ndarray): 常数项

        Returns:
            np.ndarray: 一维的 X 向量
        """
        ans = np.linalg.inv(A).dot(b)
        return ans
    
    @classmethod
    def AX_equal_0(self, A):
        """求解齐次方程 AX = 0
        Args:
            A (np.ndarray): 系数矩阵

        Returns:
            np.ndarray: 一维的 X 向量
        """
        
        def solution(U):
            # find the eigenvalues and eigenvector of U(transpose).U
            e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
            # extract the eigenvector (column) associated with the minimum eigenvalue
            return e_vecs[:, np.argmin(e_vals)] 
        
        ans = solution(A)
        return ans
            
    @classmethod
    def rank(self, A):
        """获得矩阵的迹

        Args:
            A (np.ndarray): 矩阵
        
        Returns:
            int: 迹
        """
        return np.linalg.matrix_rank(A) 


class MMPP:
    def __init__(self, max_load_internal=5) -> None:
        self.state_num = 3
        num = self.state_num
        self.trans_mat = np.array([[0. for _ in range(num)] for _ in range(num)])   # 转移概率矩阵
        self.init_dist = np.array([0. for _ in range(num)])     # 初始分布
        self.steady_dist = np.array([0. for _ in range(num)])   # 稳态分布
        
        # 这里需要控制 lam 的随机值和 r * max_load_internal 处于同一量级（简单的做法是让其均值为该数）
        r = 0.2
        # self.state_lambda = np.array([(1. * np.random.randint(0, 1000)/500. * (max_load_internal * r)) for _ in range(num)])
        self.state_lambda = np.array([0. for _ in range(num)])
        for i in range(num):
            self.state_lambda[i] = (i+1) * (max_load_internal * r) / 2.
        
        self.mean_arrival = 0.
        
        self.state = -1
        
        self.reset_params()
    
    def generate_arrivals(self):
        # 输出当前状态下到达的事件个数
        lam = self.state_lambda[self.state]
        arrivals = np.random.poisson(lam=lam)
        return arrivals
    
    def next_state(self):
        num = self.state_num
        i = self.state
        self.state = num    # 给个初值，以免 r 取到 1.
        r = np.random.randint(0, 1000) * 1. / 1000.
        now_ = 0.
        for j in range(num):
            next_ = now_ + self.trans_mat[i][j]
            if now_ <= r < next_:
                self.state = j
                break
            now_ = next_
    
    def reset_state(self):
        num = self.state_num
        self.state = num    # 给个初值，以免 r 取到 1.
        r = np.random.randint(0, 1000) * 1. / 1000.
        now_ = 0.
        for j in range(num):
            next_ = now_ + self.init_dist[j]
            if now_ <= r < next_:
                self.state = j
                break
            now_ = next_
    
    def reset_params(self):
        num = self.state_num
        # 生成转移矩阵
        rank = 0
        while rank != num:
            for i in range(num):
                p = []
                for j in range(num):
                    p.append(np.random.randint(0, 1000))
                for j in range(num):
                    self.trans_mat[i][j] = 1. * p[j] / sum(p)
            rank = LinearSolver.rank(self.trans_mat)
        # 生成初始分布
        p = []
        for j in range(num):
            p.append(np.random.randint(0, 1000))
        for j in range(num):
            self.init_dist[j] = 1. * p[j] / sum(p)
        
        self.cal_steady_dist()
        self.cal_mean_arrival()
        self.reset_state()
        
    def cal_steady_dist(self):
        num = self.state_num
        I = np.eye(num)
        A = (self.trans_mat - I).T
        X = LinearSolver.AX_equal_0(A=A)
        for j in range(num):
            self.steady_dist[j] = X[j] / sum(X)
        
        # 验证是否正确
        next_dist = self.steady_dist.dot(self.trans_mat)
        if sum(abs(next_dist - self.steady_dist)) > 0.1:
            print(f"Someting is wrong when calculating the steady states distribution! \n A: {A} \n rank of A:{LinearSolver.rank(A)};  rank of trans: {LinearSolver.rank(self.trans_mat)} \n X: {self.steady_dist} \n dot: {A.dot(self.steady_dist)}")

    def cal_mean_arrival(self):
        self.mean_arrival = 0.
        num = self.state_num
        for j in range(num):
            self.mean_arrival += self.steady_dist[j] * self.state_lambda[j]
        
                

# # test LinearSolver
# A = np.array([[7, 3, 0, 1], [0, 1, 0, -1], [1, 0, 6, -3], [1, 1, -1, -1]])
# B = np.array([8, 6, -3, 1])
# X = LinearSolver.AX_equal_b(A=A, b=B)
# print(f"AX_equal_b: {X}")    # [-3.30612245  9.28571429  1.69387755  3.28571429]

# A = np.array([[1., 1., 0., 0.], [0., 1., 0., -1.]])
# X = LinearSolver.AX_equal_0(A)
# print(f"AX_equal_0: {X}")

# test MMPP
# mmpp = MMPP()
# print(f"{mmpp.steady_dist}")