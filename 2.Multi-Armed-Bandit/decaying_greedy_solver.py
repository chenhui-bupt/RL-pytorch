from solver import BernoulliBandit
from solver import Solver
from solver import plot_results
import numpy as np


class DecayingEpsilonGreedy(Solver):
    """" epsilon随时间衰减的贪婪算法 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1. / self.total_count:  # epsilon随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)  # 随机数种子
K = 10
bandit_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))

np.random.seed(1)
decaying_eps_greedy_solver = DecayingEpsilonGreedy(bandit_arm)
decaying_eps_greedy_solver.run(5000)
print("epsilon衰减的贪婪算法的累计懊悔为：", decaying_eps_greedy_solver.regret)
plot_results([decaying_eps_greedy_solver], ["DecayingEpsilonGreedy"])
