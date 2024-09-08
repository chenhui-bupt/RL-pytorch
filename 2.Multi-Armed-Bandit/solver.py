import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 伯努利多臂老虎机，输入k表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0~1的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大获奖概率
        self.K = K

    def step(self, k):  # reward函数
        # 当玩家选择了k号拉杆后，根据拉动该老虎机的k号拉杆获得奖励的概率返回1或0
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0



class Solver:
    """ 多臂老虎机算法基本框架 """
    """ 1.根据策略选择动作；2.根据动作获得奖励；3.更新期望奖励估值；4.更新累计后悔和计数 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累计后悔
        self.actions = []  # 记录每一步的动作
        self.regrets = []  # 记录每一步的累计懊悔

    def update_regret(self, k):
        # 计算累计后悔并保存，k为本次动作选择的拉杆编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):  # 策略action函数
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


def plot_results(solvers, solver_names):
    """ 生成累计后悔随时间变化的曲线，输入solvers是一个列表， 列表每个元素是一个特定的策略 """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)  # 随机数种子
    K = 10
    bandit_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))
