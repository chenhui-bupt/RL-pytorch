from solver import BernoulliBandit
from solver import Solver
from solver import plot_results
import numpy as np


class EpsilonGreedy(Solver):
    """" 继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化所有拉杆的期望奖励估值，Q值函数
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        # 1.action
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆

        # 2.reward
        r = self.bandit.step(k)  # 得到本次动作的奖励
        # 3.更新奖励期望（增量式更新）
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)  # 随机数种子
K = 10
bandit_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

np.random.seed(0)
eps = [1e-4, 0.01, 0.1, 0.25, 0.5]
eps_greedy_solvers = [EpsilonGreedy(bandit_arm, epsilon=e) for e in eps]
eps_greedy_solver_names = ["epsilon={}".format(e) for e in eps]
for solver in eps_greedy_solvers:
    solver.run(5000)
    print("累计懊悔为:", solver.regret)

plot_results(eps_greedy_solvers, eps_greedy_solver_names)