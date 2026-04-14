"""bellman.py

创建/更新时间: 2026-04-14

内容:
	一个 4 状态网格世界的贝尔曼期望方程教学示例。

用途:
	- 展示状态价值方程 v = r + gamma * P_pi v 的矩阵写法
	- 演示如何直接解线性方程组得到状态价值
	- 演示如何通过迭代贝尔曼备份做策略评估
	- 绘制 2x2 网格、策略箭头、状态价值和收敛曲线

依赖:
	numpy, matplotlib

运行方式:
	python bellman.py
"""

from __future__ import annotations

import platform
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


StateName = str
TransitionMap = Dict[int, List[Tuple[int, float]]]


def build_demo_mrp() -> Tuple[List[StateName], np.ndarray, np.ndarray, TransitionMap]:
	"""构造与示意图一致的 4 状态 MRP。

	状态布局是一个 2x2 小网格：

		s1 | s2
		--- + ---
		s3 | s4

	这个函数返回三类对象：
	- 状态名称，方便打印和画图
	- 奖励向量 r_pi
	- 转移矩阵 P_pi

	同时返回 policy_transitions，方便在可视化里解释每个状态的转移方向。
	"""

	state_names = ["s1", "s2", "s3", "s4"]
	rewards = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)

	transition_matrix = np.array(
		[
			[0.0, 0.5, 0.5, 0.0],
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=float,
	)

	policy_transitions: TransitionMap = {
		0: [(1, 0.5), (2, 0.5)],
		1: [(3, 1.0)],
		2: [(3, 1.0)],
		3: [(3, 1.0)],
	}

	return state_names, rewards, transition_matrix, policy_transitions


def solve_bellman_equation(transition_matrix: np.ndarray, rewards: np.ndarray, gamma: float) -> np.ndarray:
	"""直接求解贝尔曼期望方程的线性形式。

	公式是：
	    (I - gamma * P_pi) v = r_pi

	其中 P_pi 是固定策略下的状态转移矩阵，r_pi 是即时奖励向量。
	这个函数适合用来验证迭代结果，也能帮助理解贝尔曼方程本质上就是一个线性方程组。

	参数:
	    transition_matrix: 状态转移矩阵 P_pi。
	    rewards: 奖励向量 r_pi。
	    gamma: 折扣因子，必须满足 0 <= gamma < 1。

	返回:
	    求解得到的状态价值向量 v_pi。
	"""

	if not 0.0 <= gamma < 1.0:
		raise ValueError("This demo expects 0 <= gamma < 1 so the value function stays finite.")

	identity = np.eye(transition_matrix.shape[0])
	system_matrix = identity - gamma * transition_matrix
	return np.linalg.solve(system_matrix, rewards)


def iterative_policy_evaluation(
	transition_matrix: np.ndarray,
	rewards: np.ndarray,
	gamma: float,
	theta: float = 1e-10,
	max_iterations: int = 10_000,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
	"""通过迭代贝尔曼备份计算状态价值。

	迭代公式为：
	    v_{k+1} = r_pi + gamma * P_pi v_k

	实现方式是从全 0 向量开始，不断把上一轮的价值代入右侧，
	直到相邻两轮的最大变化量小于 theta。

	参数:
	    transition_matrix: 状态转移矩阵 P_pi。
	    rewards: 奖励向量 r_pi。
	    gamma: 折扣因子。
	    theta: 收敛阈值。
	    max_iterations: 最大迭代次数，避免极端情况下死循环。

	返回:
	    value: 收敛后的状态价值向量。
	    history: 每一次迭代的价值向量，便于画收敛曲线。
	    deltas: 每一步的最大变化量，便于观察是否收敛。
	"""

	value = np.zeros_like(rewards, dtype=float)
	history = [value.copy()]
	deltas: List[float] = []

	for _ in range(max_iterations):
		new_value = rewards + gamma * transition_matrix @ value
		delta = float(np.max(np.abs(new_value - value)))

		history.append(new_value.copy())
		deltas.append(delta)

		value = new_value
		if delta < theta:
			break

	return value, history, deltas


def bellman_residual(
	value: np.ndarray,
	transition_matrix: np.ndarray,
	rewards: np.ndarray,
	gamma: float,
) -> np.ndarray:
	"""计算贝尔曼残差。

	残差定义为：
	    r_pi + gamma * P_pi v - v

	当 value 已经收敛到贝尔曼方程的解时，这个向量应该非常接近 0。
	它常被用来检查“当前的价值估计离真正解还有多远”。

	返回:
	    每个状态对应的残差向量。
	"""

	return rewards + gamma * transition_matrix @ value - value


def format_vector(name: str, vector: Sequence[float]) -> str:
	"""把向量格式化成便于在终端打印的字符串。

	这个辅助函数主要用于把 r_pi、v_pi 和残差输出得更整齐，
	方便你在控制台直接对照公式看结果。
	"""

	rounded = np.array(vector, dtype=float)
	return f"{name} = {np.array2string(rounded, precision=4, suppress_small=True)}"


def print_demo_summary(
	state_names: Sequence[StateName],
	rewards: np.ndarray,
	transition_matrix: np.ndarray,
	gamma: float,
	solved_values: np.ndarray,
	iter_values: np.ndarray,
) -> None:
	"""打印示例中的核心结果，帮助你在终端逐项核对。

	输出内容包括：奖励向量、转移矩阵、直接求解结果、迭代求解结果和残差。
	这部分是“看公式”和“看代码”之间的桥梁。
	"""

	np.set_printoptions(precision=4, suppress=True)

	print("Bellman expectation equation demo")
	print(f"gamma = {gamma}")
	print()
	print(format_vector("r_pi", rewards))
	print("P_pi =")
	print(transition_matrix)
	print()
	print(format_vector("v_star (linear solve)", solved_values))
	print(format_vector("v_star (iterative)", iter_values))
	print(format_vector("residual", bellman_residual(solved_values, transition_matrix, rewards, gamma)))
	print()
	for index, state_name in enumerate(state_names):
		print(f"{state_name}: v = {solved_values[index]:.6f}")


def draw_arrow(
	ax: plt.Axes,
	start: Tuple[float, float],
	end: Tuple[float, float],
	label: str,
	label_offset: Tuple[float, float],
	color: str = "crimson",
	connectionstyle: str = "arc3,rad=0.0",
) -> None:
	"""在网格上画一个带概率标签的策略箭头。

	start 和 end 用的是网格中的坐标点，label 用来标注转移概率，
	这样可以更直观地把策略转移和状态价值联系起来。
	"""

	arrow = FancyArrowPatch(
		start,
		end,
		arrowstyle="->",
		mutation_scale=18,
		linewidth=2.0,
		color=color,
		connectionstyle=connectionstyle,
	)
	ax.add_patch(arrow)

	label_x = (start[0] + end[0]) / 2.0 + label_offset[0]
	label_y = (start[1] + end[1]) / 2.0 + label_offset[1]
	ax.text(label_x, label_y, label, color=color, fontsize=10, ha="center", va="center")


def plot_gridworld(
	state_names: Sequence[StateName],
	rewards: np.ndarray,
	values: np.ndarray,
	policy_transitions: TransitionMap,
	gamma: float,
	history: Sequence[np.ndarray],
	save_path: str = "bellman_gridworld.png",
) -> None:
	"""绘制教学图：公式、网格状态价值和迭代收敛曲线。

	这个图分成三块：
	- 顶部：贝尔曼方程的文字说明
	- 左下：2x2 网格和状态价值热图
	- 右下：迭代策略评估的收敛曲线

	参数:
	    state_names: 状态名称，画在每个格子里。
	    rewards: 奖励向量，用于标注每个状态的即时奖励。
	    values: 最终求得的状态价值。
	    policy_transitions: 预留的策略转移信息，便于后续扩展成自动画箭头。
	    gamma: 折扣因子，用于标题和公式展示。
	    history: 每次迭代的价值轨迹，用来画收敛曲线。
	    save_path: 图片保存路径。
	"""

	value_grid = np.array([[values[0], values[1]], [values[2], values[3]]], dtype=float)
	vmin = float(np.min(values))
	vmax = float(np.max(values))

	fig = plt.figure(figsize=(14, 8))
	grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.1, 3.0], width_ratios=[1.1, 1.2])

	ax_text = fig.add_subplot(grid_spec[0, :])
	ax_grid = fig.add_subplot(grid_spec[1, 0])
	ax_curve = fig.add_subplot(grid_spec[1, 1])

	ax_text.axis("off")
	equation_text = (
		"Bellman expectation equation\n"
		f"v = r + gamma * P_pi v,    gamma = {gamma}\n"
		"(I - gamma * P_pi) v = r\n\n"
		f"s1: v = {rewards[0]:.1f} + gamma * (0.5 * v(s2) + 0.5 * v(s3))\n"
		f"s2: v = {rewards[1]:.1f} + gamma * v(s4)\n"
		f"s3: v = {rewards[2]:.1f} + gamma * v(s4)\n"
		f"s4: v = {rewards[3]:.1f} + gamma * v(s4)"
	)
	ax_text.text(
		0.01,
		0.95,
		equation_text,
		ha="left",
		va="top",
		fontsize=12,
		family="monospace",
		bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f6", edgecolor="#cccccc"),
	)

	# Draw the 2x2 grid with a heatmap background.
	cmap = plt.cm.viridis
	norm = plt.Normalize(vmin=vmin, vmax=vmax)
	for row in range(2):
		for col in range(2):
			idx = row * 2 + col
			rect = Rectangle((col, row), 1, 1, facecolor=cmap(norm(values[idx])), edgecolor="black", linewidth=2)
			ax_grid.add_patch(rect)

			center_x, center_y = col + 0.5, row + 0.5
			ax_grid.text(center_x, center_y - 0.18, state_names[idx], ha="center", va="center", fontsize=13, weight="bold")
			ax_grid.text(center_x, center_y + 0.02, f"V={values[idx]:.3f}", ha="center", va="center", fontsize=13, color="white", weight="bold")
			ax_grid.text(center_x, center_y + 0.24, f"r={rewards[idx]:.0f}", ha="center", va="center", fontsize=11, color="#ff3333")

	# Match the slide layout: top-left is s1, top-right is s2, bottom-left is s3, bottom-right is s4.
	ax_grid.set_xlim(0, 2)
	ax_grid.set_ylim(2, 0)
	ax_grid.set_aspect("equal")
	ax_grid.set_xticks([])
	ax_grid.set_yticks([])
	ax_grid.set_title("2x2 Gridworld: policy arrows and solved values", pad=12)

	# Policy arrows shown in the same spirit as the slide.
	draw_arrow(ax_grid, (0.68, 0.5), (1.25, 0.5), "0.5", (0.0, -0.13), color="crimson")
	draw_arrow(ax_grid, (0.5, 0.68), (0.5, 1.25), "0.5", (-0.16, 0.0), color="crimson")
	draw_arrow(ax_grid, (1.5, 0.68), (1.5, 1.25), "1.0", (0.16, 0.0), color="crimson")
	draw_arrow(ax_grid, (0.68, 1.5), (1.25, 1.5), "1.0", (0.0, 0.13), color="crimson")

	# Add a color bar for the value scale.
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	fig.colorbar(sm, ax=ax_grid, fraction=0.046, pad=0.04, label="state value")

	# Convergence plot for the iterative Bellman backup.
	history_array = np.array(history)
	iterations = np.arange(history_array.shape[0])
	for index, state_name in enumerate(state_names):
		ax_curve.plot(iterations, history_array[:, index], linewidth=2, label=state_name)

	ax_curve.axhline(values[0], linestyle="--", linewidth=1, color="gray", alpha=0.6)
	ax_curve.set_title("Iterative policy evaluation")
	ax_curve.set_xlabel("Iteration")
	ax_curve.set_ylabel("State value")
	ax_curve.grid(True, linestyle="--", alpha=0.35)
	ax_curve.legend(frameon=False)

	fig.suptitle("Bellman equation: direct solve and iterative backup", fontsize=15, weight="bold")
	fig.tight_layout(rect=[0, 0, 1, 0.96])
	fig.savefig(save_path, dpi=200, bbox_inches="tight")
	plt.show()




def main(show_plot: bool = True) -> None:
	"""主入口。

	流程是：
	1. 构造示例 MRP
	2. 直接求解贝尔曼方程
	3. 迭代求值并记录收敛过程
	4. 打印结果摘要
	5. 需要时绘制并保存图像

	参数:
	    show_plot: 是否弹出图窗。
	"""

	if platform.system() == "Windows":
		import io
		import sys

		sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

	state_names, rewards, transition_matrix, policy_transitions = build_demo_mrp()
	gamma = 0.3

	solved_values = solve_bellman_equation(transition_matrix, rewards, gamma)
	iter_values, history, deltas = iterative_policy_evaluation(transition_matrix, rewards, gamma)

	print_demo_summary(state_names, rewards, transition_matrix, gamma, solved_values, iter_values)

	print()
	print(f"Iterations until convergence: {len(deltas)}")
	print(f"Last max update: {deltas[-1]:.3e}")

	if show_plot:
		plot_gridworld(
			state_names=state_names,
			rewards=rewards,
			values=solved_values,
			policy_transitions=policy_transitions,
			gamma=gamma,
			history=history,
		)


if __name__ == "__main__":
	main()
