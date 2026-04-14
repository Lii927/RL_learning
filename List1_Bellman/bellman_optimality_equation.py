"""bellman_optimality_equation.py

创建/更新时间: 2026-04-14

内容:
	一个 5x5 网格世界的贝尔曼最优公式教学示例。

用途:
	- 计算最优状态价值 v*
	- 从 v* 提取最优策略 pi*
	- 用图形方式展示网格、奖励区和价值分布
	- 让你看到贝尔曼最优方程如何落到代码里

依赖:
	numpy, matplotlib
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


State = Tuple[int, int]
Action = int

ACTION_DELTAS: Tuple[State, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_ARROWS: Tuple[str, ...] = ("↑", "↓", "←", "→")
ACTION_LETTERS: Tuple[str, ...] = ("U", "D", "L", "R")
ACTION_PRIORITY: Tuple[int, ...] = (1, 3, 0, 2)


@dataclass(frozen=True)
class GridWorldSpec:
	"""网格世界的静态配置。

	这里的布局是根据幻灯片的示意做的一个近似版本。
	如果你想严格复现某一张课件图，只需要调整 target 和 forbidden。
	"""

	nrow: int = 5
	ncol: int = 5
	target: State = (3, 2)
	forbidden: Tuple[State, ...] = ((1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1))
	reward_target: float = 1.0
	reward_forbidden: float = -1.0
	reward_boundary: float = -1.0
	reward_normal: float = 0.0


class GridWorld:
	"""用于贝尔曼最优值迭代的简单确定性网格世界。"""

	def __init__(self, spec: GridWorldSpec):
		self.spec = spec
		self.nrow = spec.nrow
		self.ncol = spec.ncol
		self.target = spec.target
		self.forbidden = set(spec.forbidden)
		self.reward_map = np.full((self.nrow, self.ncol), spec.reward_normal, dtype=float)

		for row, col in self.forbidden:
			self.reward_map[row, col] = spec.reward_forbidden
		target_row, target_col = self.target
		self.reward_map[target_row, target_col] = spec.reward_target

	@property
	def n_states(self) -> int:
		"""返回状态总数。"""

		return self.nrow * self.ncol

	def state_to_index(self, state: State) -> int:
		"""把二维坐标转成一维状态编号。"""

		row, col = state
		return row * self.ncol + col

	def index_to_state(self, index: int) -> State:
		"""把一维状态编号转回二维坐标。"""

		return divmod(index, self.ncol)

	def cell_type(self, state: State) -> str:
		"""返回单元格类型，用于绘图着色。"""

		if state == self.target:
			return "target"
		if state in self.forbidden:
			return "forbidden"
		return "normal"

	def step(self, state: State, action: Action) -> Tuple[State, float]:
		"""执行一步转移。

		采用最直观的教学版定义：
		- 进入目标格时得到 +1 奖励，并停留在目标格
		- 进入 forbidden 格时得到 -1 奖励，但仍可继续移动
		- 撞到边界时原地不动，得到边界惩罚 -1
		- 普通格奖励为 0
		"""

		if state == self.target:
			return state, self.spec.reward_target

		row, col = state
		delta_row, delta_col = ACTION_DELTAS[action]
		next_row = row + delta_row
		next_col = col + delta_col

		if next_row < 0 or next_row >= self.nrow or next_col < 0 or next_col >= self.ncol:
			return state, self.spec.reward_boundary

		next_state = (next_row, next_col)
		return next_state, float(self.reward_map[next_row, next_col])


def build_demo_world() -> GridWorld:
	"""构造和课件风格接近的演示网格世界。"""

	return GridWorld(GridWorldSpec())


def compute_q_values(world: GridWorld, values: np.ndarray, gamma: float) -> np.ndarray:
	"""根据当前状态价值计算每个动作的 Q 值。

	这是贝尔曼最优方程右侧的逐动作形式：

		q(s, a) = r(s, a, s') + gamma * v(s')

	对每个状态取最大值，就得到下一轮的状态价值。
	"""

	q_values = np.zeros((world.n_states, len(ACTION_DELTAS)), dtype=float)
	for state_index in range(world.n_states):
		state = world.index_to_state(state_index)
		for action_index in range(len(ACTION_DELTAS)):
			next_state, reward = world.step(state, action_index)
			next_index = world.state_to_index(next_state)
			q_values[state_index, action_index] = reward + gamma * values[next_index]
	return q_values


def bellman_optimal_backup(world: GridWorld, values: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
	"""执行一次贝尔曼最优备份。

	对每个状态 s 计算：

		v_new(s) = max_a [ r(s, a, s') + gamma * v(s') ]

	这个函数就是最优贝尔曼方程在代码里的直接实现。
	"""

	q_values = compute_q_values(world, values, gamma)
	new_values = np.max(q_values, axis=1)
	return new_values, q_values


def value_iteration(
	world: GridWorld,
	gamma: float,
	theta: float = 1e-10,
	max_iterations: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
	"""使用价值迭代求解最优状态价值。

	返回值包含：
	- 最终状态价值
	- 最终 Q 值表
	- 每次迭代的价值历史，便于画收敛曲线
	- 每次迭代的最大变化量，便于观察是否收敛
	"""

	values = np.zeros(world.n_states, dtype=float)
	history = [values.copy()]
	deltas: List[float] = []
	q_values = compute_q_values(world, values, gamma)

	for _ in range(max_iterations):
		new_values, q_values = bellman_optimal_backup(world, values, gamma)
		delta = float(np.max(np.abs(new_values - values)))
		history.append(new_values.copy())
		deltas.append(delta)
		values = new_values
		if delta < theta:
			break

	q_values = compute_q_values(world, values, gamma)
	return values, q_values, history, deltas


def greedy_action_from_q(world: GridWorld, state: State, q_row: np.ndarray) -> int:
	"""从 Q 值中取出一个稳定的贪心动作。

	如果多个动作并列最优，会优先选择更靠近目标的动作，
	再按一个固定顺序打破平局，保证图上的箭头更稳定。
	"""

	best_value = float(np.max(q_row))
	candidate_actions = np.flatnonzero(np.isclose(q_row, best_value, atol=1e-12))
	if candidate_actions.size == 1:
		return int(candidate_actions[0])

	target_row, target_col = world.target

	def tie_key(action_index: int) -> Tuple[int, int]:
		next_state, _ = world.step(state, action_index)
		next_row, next_col = next_state
		manhattan_distance = abs(next_row - target_row) + abs(next_col - target_col)
		return manhattan_distance, ACTION_PRIORITY[action_index]

	return int(min((int(action) for action in candidate_actions), key=tie_key))


def extract_greedy_policy(world: GridWorld, q_values: np.ndarray) -> np.ndarray:
	"""把 Q 表转成动作策略。"""

	policy = np.zeros(world.n_states, dtype=int)
	for state_index in range(world.n_states):
		state = world.index_to_state(state_index)
		policy[state_index] = greedy_action_from_q(world, state, q_values[state_index])
	return policy


def state_value_grid(world: GridWorld, values: np.ndarray) -> np.ndarray:
	"""把一维价值向量 reshape 成二维网格。"""

	return values.reshape(world.nrow, world.ncol)


def print_summary(world: GridWorld, values: np.ndarray, policy: np.ndarray, gamma: float) -> None:
	"""在终端里打印最优价值和最优策略，方便对照公式。"""

	value_grid = state_value_grid(world, values)
	print("Bellman optimality equation demo")
	print(f"gamma = {gamma}")
	print(f"target = {world.target}, forbidden = {sorted(world.forbidden)}")
	print()
	print("Optimal state values:")
	print(np.array2string(np.round(value_grid, 1), precision=1, suppress_small=True))
	print()
	print("Greedy policy (U/D/L/R):")
	for row in range(world.nrow):
		symbols = []
		for col in range(world.ncol):
			state = (row, col)
			if state == world.target:
				symbols.append("T")
			else:
				symbols.append(ACTION_LETTERS[int(policy[world.state_to_index(state)])])
		print(" ".join(symbols))


def cell_face_color(world: GridWorld, state: State) -> str:
	"""根据单元格类型返回底色。"""

	cell_type = world.cell_type(state)
	if cell_type == "target":
		return "#76d7ea"
	if cell_type == "forbidden":
		return "#f4b84f"
	return "#ffffff"


def draw_grid_panel(
	ax: plt.Axes,
	world: GridWorld,
	values: np.ndarray | None = None,
	policy: np.ndarray | None = None,
	title: str = "",
) -> None:
	"""在单个坐标轴上绘制网格。

	如果传入 policy，就在格子里画动作箭头；
	如果传入 values，就在格子里画数值。
	"""

	ax.set_xlim(0, world.ncol)
	ax.set_ylim(world.nrow, 0)
	ax.set_aspect("equal")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(title, fontsize=13, pad=10)

	for row in range(world.nrow):
		for col in range(world.ncol):
			state = (row, col)
			face_color = cell_face_color(world, state)
			rect = Rectangle((col, row), 1, 1, facecolor=face_color, edgecolor="#444444", linewidth=1.8)
			ax.add_patch(rect)

			if values is not None:
				state_index = world.state_to_index(state)
				ax.text(
					col + 0.5,
					row + 0.5,
					f"{values[state_index]:.1f}",
					ha="center",
					va="center",
					fontsize=13,
					color="#222222" if world.cell_type(state) != "target" else "#111111",
					weight="bold",
				)

			if policy is not None and state != world.target:
				state_index = world.state_to_index(state)
				arrow = ACTION_ARROWS[int(policy[state_index])]
				ax.text(
					col + 0.5,
					row + 0.24,
					arrow,
					ha="center",
					va="center",
					fontsize=18,
					color="#2f7d32",
					weight="bold",
				)

			if state == world.target:
				ax.text(col + 0.5, row + 0.75, "+1", ha="center", va="center", fontsize=10, color="#a11b1b")
			elif state in world.forbidden:
				ax.text(col + 0.5, row + 0.75, "-1", ha="center", va="center", fontsize=10, color="#a11b1b")

	for col in range(world.ncol):
		ax.text(col + 0.5, -0.18, str(col + 1), ha="center", va="center", fontsize=10, color="#555555")
	for row in range(world.nrow):
		ax.text(-0.18, row + 0.5, str(row + 1), ha="center", va="center", fontsize=10, color="#555555")


def plot_results(
	world: GridWorld,
	values: np.ndarray,
	policy: np.ndarray,
	gamma: float,
	history: Sequence[np.ndarray],
	save_path: str = "bellman_optimality_equation.png",
	show: bool = True,
) -> None:
	"""绘制公式、最优策略图、最优价值图和收敛曲线。"""

	fig = plt.figure(figsize=(15, 9))
	grid_spec = fig.add_gridspec(2, 2, height_ratios=[0.95, 3.1], width_ratios=[1.05, 1.25])

	ax_text = fig.add_subplot(grid_spec[0, :])
	ax_policy = fig.add_subplot(grid_spec[1, 0])
	ax_values = fig.add_subplot(grid_spec[1, 1])

	ax_text.axis("off")
	formula_text = (
		"Bellman optimality equation\n"
		"v*(s) = max_a E[r(s, a, s') + gamma * v*(s')]\n"
		"Value iteration: v_{k+1}(s) = max_a [r(s, a, s') + gamma * v_k(s')]\n"
		f"r_boundary = r_forbidden = -1, r_target = 1, gamma = {gamma}"
	)
	ax_text.text(
		0.01,
		0.95,
		formula_text,
		ha="left",
		va="top",
		fontsize=13,
		family="monospace",
		bbox=dict(boxstyle="round,pad=0.55", facecolor="#f6f6f6", edgecolor="#cfcfcf"),
	)

	draw_grid_panel(ax_policy, world, policy=policy, title="Optimal policy")
	draw_grid_panel(ax_values, world, values=values, title="Optimal state values")

	# 收敛曲线
	history_array = np.array(history)
	iterations = np.arange(history_array.shape[0])
	ax_curve = ax_values.inset_axes([0.58, 0.03, 0.38, 0.35])
	for state_index in range(world.n_states):
		ax_curve.plot(iterations, history_array[:, state_index], linewidth=1.3, alpha=0.85)
	ax_curve.set_title("Convergence", fontsize=10)
	ax_curve.set_xlabel("Iter", fontsize=9)
	ax_curve.set_ylabel("V", fontsize=9)
	ax_curve.tick_params(labelsize=8)
	ax_curve.grid(True, linestyle="--", alpha=0.3)

	fig.text(0.5, 0.035, "The optimal policy dares to take risks: entering forbidden areas!", ha="center", fontsize=12)
	fig.text(0.5, 0.012, "Adjust TARGET and FORBIDDEN in the spec to match a different slide exactly.", ha="center", fontsize=9, color="#666666")

	fig.tight_layout(rect=[0, 0.04, 1, 0.94])
	fig.savefig(save_path, dpi=220, bbox_inches="tight")
	if show:
		plt.show()
	plt.close(fig)


def solve_for_gamma(world: GridWorld, gamma: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
	"""在给定 gamma 下求解最优价值和最优策略。"""

	values, q_values, history, deltas = value_iteration(world, gamma)
	policy = extract_greedy_policy(world, q_values)
	return values, policy, history, deltas


def print_gamma_summary(world: GridWorld, gamma: float, values: np.ndarray, deltas: Sequence[float]) -> None:
	"""打印单个 gamma 的简要结果。"""

	print(f"gamma = {gamma}")
	print(f"iterations = {len(deltas)}")
	print(f"last max update = {deltas[-1]:.3e}")
	print(np.array2string(np.round(state_value_grid(world, values), 1), precision=1, suppress_small=True))


def plot_gamma_comparison(
	world: GridWorld,
	gamma_values: Sequence[float],
	save_path: str = "bellman_optimality_equation_comparison.png",
	show: bool = True,
) -> None:
	"""并排比较多个 gamma 的最优策略和最优状态价值。"""

	gamma_values = tuple(gamma_values)
	if not gamma_values:
		raise ValueError("gamma_values cannot be empty")

	results = []
	for gamma in gamma_values:
		values, policy, history, deltas = solve_for_gamma(world, gamma)
		results.append((gamma, values, policy, history, deltas))

	fig = plt.figure(figsize=(16, 4.5 * len(results) + 1.8))
	grid_spec = fig.add_gridspec(
		len(results),
		2,
		height_ratios=[1.0] * len(results),
		width_ratios=[1.05, 1.25],
		hspace=0.35,
		wspace=0.18,
	)

	fig.suptitle("Discount factor comparison: larger gamma looks farther ahead", fontsize=15, weight="bold")

	for row, (gamma, values, policy, _history, deltas) in enumerate(results):
		ax_policy = fig.add_subplot(grid_spec[row, 0])
		ax_values = fig.add_subplot(grid_spec[row, 1])

		draw_grid_panel(ax_policy, world, policy=policy, title=f"Optimal policy (gamma={gamma})")
		draw_grid_panel(ax_values, world, values=values, title=f"Optimal state values (gamma={gamma})")

		ax_policy.text(
			0.5,
			-0.12,
			f"iterations = {len(deltas)}",
			transform=ax_policy.transAxes,
			ha="center",
			va="top",
			fontsize=9,
			color="#555555",
		)
		ax_values.text(
			0.5,
			-0.12,
			f"last update = {deltas[-1]:.1e}",
			transform=ax_values.transAxes,
			ha="center",
			va="top",
			fontsize=9,
			color="#555555",
		)

	fig.text(
		0.5,
		0.02,
		"Higher gamma weights future reward more; lower gamma is more short-sighted.",
		ha="center",
		fontsize=11,
	)

	fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.45, wspace=0.18)
	fig.savefig(save_path, dpi=220, bbox_inches="tight")
	if show:
		plt.show()
	plt.close(fig)


def _ensure_utf8_stdout() -> None:
	"""在 Windows 下尽量把 stdout 切到 UTF-8，避免中文输出乱码。"""

	if platform.system() != "Windows":
		return

	import sys

	stdout = sys.stdout
	if getattr(stdout, "encoding", "").lower() == "utf-8":
		return

	reconfigure = getattr(stdout, "reconfigure", None)
	if callable(reconfigure):
		reconfigure(encoding="utf-8")
		return

	import io

	sys.stdout = io.TextIOWrapper(stdout.buffer, encoding="utf-8", write_through=True)


def main(
	gamma_values: Sequence[float] | None = None,
	show_plot: bool = True,
	save_path: str = "bellman_optimality_equation_comparison.png",
) -> None:
	"""主入口：默认并排比较多个 gamma，也支持单个 gamma。"""

	_ensure_utf8_stdout()

	world = build_demo_world()
	if gamma_values is None:
		gamma_values = (0.9, 0.5, 0.3)
	gamma_values = tuple(gamma_values)

	if len(gamma_values) == 1:
		gamma = gamma_values[0]
		values, policy, history, deltas = solve_for_gamma(world, gamma)
		print_summary(world, values, policy, gamma)
		print()
		print(f"Iterations until convergence: {len(deltas)}")
		print(f"Last max update: {deltas[-1]:.3e}")
		plot_results(world, values, policy, gamma, history, save_path=save_path, show=show_plot)
		return

	comparison_results = []
	for gamma in gamma_values:
		values, policy, history, deltas = solve_for_gamma(world, gamma)
		comparison_results.append((gamma, values, policy, history, deltas))

	print("Discount factor comparison")
	for gamma, values, _policy, _history, deltas in comparison_results:
		print_gamma_summary(world, gamma, values, deltas)
		print()

	plot_gamma_comparison(world, gamma_values, save_path=save_path, show=show_plot)


if __name__ == "__main__":
	main()
