from solver import BernoulliBandit
from solver import Solver
from solver import plot_results
import numpy as np


class ThompsonSampling(Solver):
    """ 汤普森采样算法 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照beta分布采样一组奖励样本
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r  # 更新beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新beta分布的第二个参数
        return k


np.random.seed(1)  # 随机数种子
K = 10
bandit_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))

np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_arm)
thompson_sampling_solver.run(5000)
print("汤普森采样算法的累计懊悔为：", thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])


"""
结论：
1. epsilon贪婪算法的累计懊悔是随时间线性增长的；
2. epsilon衰减贪婪算法、上置信界算法、汤普森采样算法的累计懊悔是随时间次线性增长的（以对数形式增长）
"""