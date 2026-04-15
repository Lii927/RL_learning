"""policy_iteration.py

策略迭代算法的具体实现。

这个脚本对应课件里的 5x5 网格世界复杂示例：

	r_boundary = -1, r_forbidden = -10, r_target = 1, gamma = 0.3

网格布局与课件中的示意一致，target 与 forbidden 的位置保持不变。
本版本使用 5 个动作：上、下、左、右、停留。

算法流程是：
1. 用一个初始策略（这里默认全 stay）开始
2. 做策略评估，解出当前策略对应的状态价值 v_pi
3. 做策略改进，用 q(s, a) 贪心更新策略
4. 重复上述过程直到策略不再变化

图里会展示最终收敛结果和全部策略迭代阶段的 value 收敛过程。
"""

from __future__ import annotations

import io
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


State = Tuple[int, int]
Action = int

ACTION_DELTAS: Tuple[State, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_ARROWS: Tuple[str, ...] = ("↑", "↓", "←", "→", "○")
ACTION_LETTERS: Tuple[str, ...] = ("U", "D", "L", "R", "S")
ACTION_PRIORITY: Tuple[int, ...] = (1, 3, 0, 2, 4)


@dataclass(frozen=True)
class GridWorldSpec:
	nrow: int = 5
	ncol: int = 5
	target: State = (3, 2)
	forbidden: Tuple[State, ...] = ((1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1))
	reward_target: float = 1.0
	reward_forbidden: float = -10.0
	reward_boundary: float = -1.0
	reward_normal: float = 0.0


class GridWorld:
	"""用于策略迭代的确定性网格世界。"""

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
		"""把二维坐标转成一维编号。"""

		row, col = state
		return row * self.ncol + col

	def index_to_state(self, index: int) -> State:
		"""把一维编号转回二维坐标。"""

		return divmod(index, self.ncol)

	def cell_type(self, state: State) -> str:
		"""返回单元格类型，用于绘图上色。"""

		if state == self.target:
			return "target"
		if state in self.forbidden:
			return "forbidden"
		return "normal"

	def step(self, state: State, action: Action) -> Tuple[State, float]:
		"""执行一次确定性转移。

		规则与课件一致：
		- 进入 target：奖励 +1，并停留在 target
		- 进入 forbidden：奖励 -10
		- 撞到边界：原地不动，奖励 -1
		- 其余普通格：奖励 0
		- 动作 stay：停留在原地，奖励由当前格类型决定
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
		reward = float(self.reward_map[next_row, next_col])
		return next_state, reward


@dataclass(frozen=True)
class PolicyIterationStage:
	"""记录一次策略迭代中的策略和值。"""

	iteration: int
	policy: np.ndarray
	values: np.ndarray
	q_values: np.ndarray


def build_demo_world() -> GridWorld:
	"""构造课件里的演示网格世界。"""

	return GridWorld(GridWorldSpec())


def build_initial_policy(world: GridWorld) -> np.ndarray:
	"""构造初始策略：全 stay。

	这个初始策略能直接复现课件里 π0 / vπ0 的数值结构：
	普通格为 0，forbidden 为 -100，target 为 10。
	"""

	return np.full(world.n_states, 4, dtype=int)


def policy_transition_and_reward(world: GridWorld, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""根据固定策略构造状态转移矩阵和即时奖励向量。"""

	transition_matrix = np.zeros((world.n_states, world.n_states), dtype=float)
	rewards = np.zeros(world.n_states, dtype=float)

	for state_index in range(world.n_states):
		state = world.index_to_state(state_index)
		action = int(policy[state_index])
		next_state, reward = world.step(state, action)
		next_index = world.state_to_index(next_state)
		transition_matrix[state_index, next_index] = 1.0
		rewards[state_index] = reward

	return transition_matrix, rewards


def policy_evaluation(world: GridWorld, policy: np.ndarray, gamma: float) -> np.ndarray:
	"""精确求解当前策略对应的状态价值。"""

	transition_matrix, rewards = policy_transition_and_reward(world, policy)
	identity = np.eye(world.n_states)
	return np.linalg.solve(identity - gamma * transition_matrix, rewards)


def compute_q_values(world: GridWorld, values: np.ndarray, gamma: float) -> np.ndarray:
	"""根据状态价值计算 q(s, a)。"""

	q_values = np.zeros((world.n_states, len(ACTION_DELTAS)), dtype=float)
	for state_index in range(world.n_states):
		state = world.index_to_state(state_index)
		for action_index in range(len(ACTION_DELTAS)):
			next_state, reward = world.step(state, action_index)
			next_index = world.state_to_index(next_state)
			q_values[state_index, action_index] = reward + gamma * values[next_index]
	return q_values


def greedy_action_from_q(world: GridWorld, state: State, q_row: np.ndarray) -> int:
	"""从一行 q 值里取出一个稳定的贪心动作。"""

	if state == world.target:
		return 4

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
	"""把 q 表转成策略。"""

	policy = np.zeros(world.n_states, dtype=int)
	for state_index in range(world.n_states):
		state = world.index_to_state(state_index)
		policy[state_index] = greedy_action_from_q(world, state, q_values[state_index])
	return policy


def policy_iteration(
	world: GridWorld,
	gamma: float,
	initial_policy: np.ndarray | None = None,
	max_iterations: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PolicyIterationStage]]:
	"""运行完整的策略迭代。

	返回：
	- 最优策略
	- 最优状态价值
	- 最终 q 表
	- 中间阶段记录
	"""

	policy = build_initial_policy(world) if initial_policy is None else np.array(initial_policy, dtype=int).copy()
	stages: List[PolicyIterationStage] = []

	for iteration in range(max_iterations):
		values = policy_evaluation(world, policy, gamma)
		q_values = compute_q_values(world, values, gamma)
		stages.append(
			PolicyIterationStage(
				iteration=iteration,
				policy=policy.copy(),
				values=values.copy(),
				q_values=q_values.copy(),
			)
		)

		new_policy = extract_greedy_policy(world, q_values)
		if np.array_equal(new_policy, policy):
			return policy, values, q_values, stages
		policy = new_policy

	raise RuntimeError("Policy iteration did not converge within max_iterations.")


def _format_cell(value: float) -> str:
	"""格式化表格单元格里的数值。"""

	text = f"{value:.1f}"
	return text[:-2] if text.endswith(".0") else text


def format_value_grid(world: GridWorld, values: np.ndarray) -> str:
	"""把状态价值打印成 5x5 网格。"""

	grid = values.reshape(world.nrow, world.ncol)
	return np.array2string(np.round(grid, 1), precision=1, suppress_small=True)


def format_stage_deltas(stages: Sequence[PolicyIterationStage]) -> str:
	"""把每次策略改进带来的 value 最大变化量格式化成多行文本。"""

	deltas: List[str] = []
	for index in range(1, len(stages)):
		previous_values = stages[index - 1].values
		current_values = stages[index].values
		delta = float(np.max(np.abs(current_values - previous_values)))
		deltas.append(f"step {index}: max |ΔV| = {delta:.6f}")
	return "\n".join(deltas)


def format_policy_grid(world: GridWorld, policy: np.ndarray) -> str:
	"""把策略打印成 5x5 网格。"""

	rows: List[str] = []
	for row in range(world.nrow):
		cells: List[str] = []
		for col in range(world.ncol):
			state = (row, col)
			if state == world.target:
				cells.append("T")
			elif state in world.forbidden:
				cells.append("F")
			else:
				cells.append(ACTION_LETTERS[int(policy[world.state_to_index(state)])])
		rows.append(" ".join(f"{cell:^3}" for cell in cells))
	return "\n".join(rows)


def cell_face_color(world: GridWorld, state: State) -> str:
	"""根据单元格类型返回底色。"""

	cell_type = world.cell_type(state)
	if cell_type == "target":
		return "#76d7ea"
	if cell_type == "forbidden":
		return "#f4b84f"
	return "#ffffff"


def draw_arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = "#2f7d32") -> None:
	"""在网格上画一个小箭头。"""

	arrow = FancyArrowPatch(
		start,
		end,
		arrowstyle="->",
		mutation_scale=16,
		linewidth=2.0,
		color=color,
		zorder=5,
	)
	ax.add_patch(arrow)


def draw_policy_panel(ax: plt.Axes, world: GridWorld, policy: np.ndarray, title: str = "") -> None:
	"""绘制策略面板。"""

	ax.set_xlim(0, world.ncol)
	ax.set_ylim(world.nrow, 0)
	ax.set_aspect("equal")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(title, fontsize=13, pad=10)

	for row in range(world.nrow):
		for col in range(world.ncol):
			state = (row, col)
			rect = Rectangle((col, row), 1, 1, facecolor=cell_face_color(world, state), edgecolor="#444444", linewidth=1.6)
			ax.add_patch(rect)

			if state == world.target:
				ax.text(col + 0.5, row + 0.5, "T", ha="center", va="center", fontsize=18, color="#2f7d32", weight="bold")
				continue

			action = int(policy[world.state_to_index(state)])
			arrow = ACTION_ARROWS[action]
			top_y = row + 0.22
			center_y = row + 0.5
			if action == 4:
				ax.text(col + 0.5, center_y, arrow, ha="center", va="center", fontsize=18, color="#2f7d32", weight="bold")
			else:
				delta_row, delta_col = ACTION_DELTAS[action]
				start = (col + 0.5, center_y)
				end = (col + 0.5 + 0.22 * delta_col, center_y + 0.22 * delta_row)
				draw_arrow(ax, start, end)

	for col in range(world.ncol):
		ax.text(col + 0.5, -0.18, str(col + 1), ha="center", va="center", fontsize=9, color="#666666")
	for row in range(world.nrow):
		ax.text(-0.18, row + 0.5, str(row + 1), ha="center", va="center", fontsize=9, color="#666666")


def draw_value_panel(ax: plt.Axes, world: GridWorld, values: np.ndarray, title: str = "") -> None:
	"""绘制状态价值面板。"""

	ax.set_xlim(0, world.ncol)
	ax.set_ylim(world.nrow, 0)
	ax.set_aspect("equal")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(title, fontsize=13, pad=10)

	for row in range(world.nrow):
		for col in range(world.ncol):
			state = (row, col)
			rect = Rectangle((col, row), 1, 1, facecolor=cell_face_color(world, state), edgecolor="#444444", linewidth=1.6)
			ax.add_patch(rect)

			state_index = world.state_to_index(state)
			ax.text(
				col + 0.5,
				row + 0.5,
				f"{values[state_index]:.1f}",
				ha="center",
				va="center",
				fontsize=13,
				weight="bold",
				color="#222222",
			)

	for col in range(world.ncol):
		ax.text(col + 0.5, -0.18, str(col + 1), ha="center", va="center", fontsize=9, color="#666666")
	for row in range(world.nrow):
		ax.text(-0.18, row + 0.5, str(row + 1), ha="center", va="center", fontsize=9, color="#666666")


def plot_policy_iteration_convergence(
	world: GridWorld,
	stages: Sequence[PolicyIterationStage],
	gamma: float,
	filename: str = "policy_iteration_convergence.png",
	show: bool = True,
) -> None:
	"""把最终策略和整个 value 收敛过程可视化出来。"""

	if not stages:
		raise ValueError("stages cannot be empty")

	history = np.stack([stage.values for stage in stages], axis=0)
	iteration_indices = np.arange(history.shape[0])
	if history.shape[0] > 1:
		value_deltas = np.array([np.max(np.abs(history[index] - history[index - 1])) for index in range(1, history.shape[0])])
	else:
		value_deltas = np.array([])

	fig = plt.figure(figsize=(19, 8.8))
	outer = fig.add_gridspec(2, 3, height_ratios=[0.22, 1.0], width_ratios=[1.0, 1.0, 1.35], hspace=0.28, wspace=0.22)
	fig.suptitle("Policy iteration algorithm - convergence result", fontsize=18, weight="bold", y=0.98)

	ax_text = fig.add_subplot(outer[0, :])
	ax_text.axis("off")
	ax_text.text(
		0.01,
		0.72,
		f"Setting: r_boundary = -1, r_forbidden = -10, r_target = 1, γ = {gamma}.",
		ha="left",
		va="center",
		fontsize=13,
	)
	ax_text.text(
		0.01,
		0.20,
		f"Policy iteration converged after {len(stages) - 1} policy improvements; the right panel shows the value convergence history.",
		ha="left",
		va="center",
		fontsize=12,
	)

	final_stage = stages[-1]
	ax_policy = fig.add_subplot(outer[1, 0])
	ax_values = fig.add_subplot(outer[1, 1])
	convergence_spec = outer[1, 2].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.26)
	ax_delta = fig.add_subplot(convergence_spec[0, 0])
	ax_heat = fig.add_subplot(convergence_spec[1, 0])

	draw_policy_panel(ax_policy, world, final_stage.policy, title="Optimal policy")
	draw_value_panel(ax_values, world, final_stage.values, title="Optimal state values")

	if value_deltas.size > 0:
		ax_delta.plot(np.arange(1, len(stages)), value_deltas, marker="o", linewidth=2.2, color="#c0392b")
		ax_delta.set_yscale("log")
	ax_delta.set_title("Max value change per policy improvement", fontsize=12)
	ax_delta.set_xlabel("Policy improvement step")
	ax_delta.set_ylabel("max |ΔV|")
	ax_delta.grid(True, linestyle="--", alpha=0.35)
	if value_deltas.size > 0:
		ax_delta.set_xlim(1, len(stages) - 1)

	state_labels = [f"s{index + 1}" for index in range(history.shape[1])]
	heat = ax_heat.imshow(history.T, aspect="auto", origin="lower", cmap="YlGnBu")
	ax_heat.set_title("State value convergence history", fontsize=12)
	ax_heat.set_xlabel("Policy iteration step")
	ax_heat.set_ylabel("State")
	ax_heat.set_xticks(np.arange(history.shape[0]))
	ax_heat.set_xticklabels([str(index) for index in iteration_indices], fontsize=8)
	ax_heat.set_yticks(np.arange(history.shape[1]))
	ax_heat.set_yticklabels(state_labels, fontsize=7)
	ax_heat.set_ylim(-0.5, history.shape[1] - 0.5)
	for row in range(world.nrow + 1):
		ax_heat.axhline(row * world.ncol - 0.5, color="white", linewidth=0.6, alpha=0.7)
	fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04, label="Value")

	fig.text(
		0.5,
		0.02,
		"State order in the heatmap is row-major: top-left to bottom-right.",
		ha="center",
		fontsize=10,
		color="#666666",
	)

	fig.savefig(filename, dpi=220, bbox_inches="tight")
	backend = plt.get_backend().lower()
	if show and "agg" not in backend:
		plt.show()
	plt.close(fig)


def print_final_summary(world: GridWorld, policy: np.ndarray, values: np.ndarray, gamma: float, stages: Sequence[PolicyIterationStage]) -> None:
	"""打印最终收敛结果。"""

	print("Policy iteration demo")
	print(f"gamma = {gamma}")
	print(f"target = {world.target}, forbidden = {sorted(world.forbidden)}")
	print(f"stages = {len(stages)}")
	print()
	print("Max |ΔV| per policy improvement:")
	print(format_stage_deltas(stages))
	print()
	print("Optimal state values:")
	print(format_value_grid(world, values))
	print()
	print("Optimal policy (U/D/L/R/S):")
	print(format_policy_grid(world, policy))



def _ensure_utf8_stdout() -> None:
	"""在 Windows 下尽量把 stdout 切到 UTF-8，避免中文输出乱码。"""

	if platform.system() != "Windows":
		return

	stdout = sys.stdout
	if getattr(stdout, "encoding", "").lower() == "utf-8":
		return

	reconfigure = getattr(stdout, "reconfigure", None)
	if callable(reconfigure):
		reconfigure(encoding="utf-8")
		return

	sys.stdout = io.TextIOWrapper(stdout.buffer, encoding="utf-8", write_through=True)


def main(gamma: float = 0.3, show_plot: bool = True) -> None:
	"""主入口：运行策略迭代并生成课件风格图表。"""

	_ensure_utf8_stdout()

	world = build_demo_world()
	final_policy, final_values, final_q_values, stages = policy_iteration(world, gamma=gamma)

	print_final_summary(world, final_policy, final_values, gamma, stages)

	output_path = Path(__file__).resolve().with_name("policy_iteration_convergence.png")
	plot_policy_iteration_convergence(world, stages, gamma=gamma, filename=str(output_path), show=show_plot)
	print()
	print(f"Saved figure: {output_path}")


if __name__ == "__main__":
	main()
