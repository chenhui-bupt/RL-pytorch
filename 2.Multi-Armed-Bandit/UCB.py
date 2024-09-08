from solver import BernoulliBandit
from solver import Solver
from solver import plot_results
import numpy as np


class UCB(Solver):
    """ 上置信界法(upper confidence bound) """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) /
                                                   (2 * (self.counts + 1)))
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)  # 随机数种子
K = 10
bandit_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))

np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_arm, coef)
UCB_solver.run(5000)
print("上置信界算法的累计懊悔为：", UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
