import numpy as np
from tqdm import tqdm
import platform
import matplotlib.pyplot as plt

# ==========================================
# 1. 环境定义
# ==========================================
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        self.x = min(self.ncol - 1, max(0, self.x + self.change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + self.change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

# ==========================================
# 2. SARSA 算法类
# ==========================================
class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [i for i in range(self.n_action) if self.Q_table[state, i] == Q_max]
        return np.random.choice(a)

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

# ==========================================
# 3. 箭头策略可视化函数
# ==========================================
def print_agent_arrows(agent, env, action_arrows, disaster=[], end=[]):
    print('\n最终决策路线图 (箭头指示最佳移动方向):')
    cliff_sign = ' ☠ ' 
    goal_sign  = ' 🚩 ' 

    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            if state in disaster:
                print(cliff_sign, end=' ')
            elif state in end:
                print(goal_sign, end=' ')
            else:
                a = agent.best_action(state)
                arrow = action_arrows[a]
                print(f' {arrow} ', end=' ')
        print() 

# ==========================================
# 4. 滑动平均滤波函数 (用于画平滑曲线)
# ==========================================
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# ==========================================
# 5. 训练主循环 & 画图
# ==========================================
if __name__ == "__main__":
    if platform.system() == "Windows":
        import sys, io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 500
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    
    # 恢复用于记录分数的列表
    return_list = [] 
    
    print("开始训练 SARSA 智能体...")
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    
                    episode_return += reward # 累加当前回合的分数
                    agent.update(state, action, reward, next_state, next_action)
                    
                    state = next_state
                    action = next_action
                
                # 回合结束，记录分数
                return_list.append(episode_return)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    # --- 打印文字版路线 ---
    action_arrows = [' ↑ ', ' ↓ ', ' ← ', ' → ']
    disaster_list = list(range(37, 47))
    end_list = [47]
    print('\nSARSA算法最终收敛得到的安全策略路径如下：')
    print_agent_arrows(agent, env, action_arrows, disaster_list, end_list)

    # --- 绘制价值收敛图 ---
    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 9) 
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, mv_return, color='b')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.title('Sarsa on Cliff Walking (Smoothed)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show() # 弹窗显示图表