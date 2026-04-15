"""value_iteration.py

值迭代算法的具体实现。

这个脚本对应课件里的 2x2 网格世界：

	s1 | s2
	--- + ---
	s3 | s4

本版本不展示符号公式，直接计算并可视化前两次更新得到的数值 q-table：
- k = 0: 由 V0 = [0, 0, 0, 0] 计算 q-table
- k = 1: 由 V1 = max_a q0(s, a) 计算 q-table

这样可以直接看到值是如何在两次备份后传播的。
"""

from __future__ import annotations

import io
import platform
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


State = int
Action = int

STATE_NAMES: Tuple[str, ...] = ("s1", "s2", "s3", "s4")
ACTION_NAMES: Tuple[str, ...] = ("a1", "a2", "a3", "a4", "a5")
ACTION_SYMBOLS: Tuple[str, ...] = ("↑", "→", "↓", "←", "·")
ACTION_DELTAS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
ACTION_COLORS: Tuple[str, ...] = ("#2f7d32", "#2f7d32", "#2f7d32", "#2f7d32", "#6c757d")


class MiniGridWorld:
	"""一个和课件一致的 2x2 教学网格世界。"""

	def __init__(self) -> None:
		self.nrow = 2
		self.ncol = 2
		self.n_states = 4
		self.n_actions = 5
		self.target = (1, 1)
		self.forbidden = {(0, 1)}

		# s1, s2, s3, s4 的二维坐标。
		self._state_positions: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
		self._position_to_state = {position: index for index, position in enumerate(self._state_positions)}

		# reward_map 体现了课件中的奖励设置：
		# - normal: 0
		# - forbidden: -1
		# - target: 1
		self.reward_map = np.array(
			[
				[0.0, -1.0],
				[0.0, 1.0],
			],
			dtype=float,
		)

		# 5 个动作：上、右、下、左、停留。
		self._action_deltas = ACTION_DELTAS

	def index_to_state(self, index: int) -> Tuple[int, int]:
		"""把状态编号转成二维坐标。"""

		return self._state_positions[index]

	def state_to_index(self, state: Tuple[int, int]) -> int:
		"""把二维坐标转成状态编号。"""

		return self._position_to_state[state]

	def cell_type(self, state: Tuple[int, int]) -> str:
		"""返回单元格类型，用于绘图着色。"""

		if state == self.target:
			return "target"
		if state in self.forbidden:
			return "forbidden"
		return "normal"

	def step(self, state: State, action: Action) -> Tuple[State, float]:
		"""执行一次确定性转移。

		奖励规则按照课件设定：
		- 进入 forbidden 格得到 -1
		- 进入 target 格得到 +1
		- 进入普通格得到 0
		- 撞到边界时留在原地，得到 -1
		"""

		row, col = self.index_to_state(state)
		delta_row, delta_col = self._action_deltas[action]
		next_row = row + delta_row
		next_col = col + delta_col

		if next_row < 0 or next_row >= self.nrow or next_col < 0 or next_col >= self.ncol:
			return state, -1.0

		next_state = self.state_to_index((next_row, next_col))
		reward = float(self.reward_map[next_row, next_col])
		return next_state, reward

	def q_expression(self, state: State, action: Action) -> str:
		"""返回课件风格的 q(s, a) 文字表达式。"""

		next_state, reward = self.step(state, action)
		reward_text = _format_reward(reward)
		return f"{reward_text} + γv({STATE_NAMES[next_state]})"


def _format_reward(value: float) -> str:
	"""把奖励格式化成更适合课堂展示的写法。"""

	if float(value).is_integer():
		return str(int(value))
	return f"{value:g}"


def _format_cell(value: float) -> str:
	"""把 q-table 的数值格式化成适合表格显示的文本。"""

	text = f"{value:.1f}"
	return text[:-2] if text.endswith(".0") else text


def compute_first_two_q_tables(world: MiniGridWorld, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""直接算出 k=0 和 k=1 的 q-table。

	返回：
	- V0
	- Q0
	- V1
	- Q1
	"""

	v0 = np.zeros(world.n_states, dtype=float)
	q0 = build_q_table(world, v0, gamma)
	v1 = np.max(q0, axis=1)
	q1 = build_q_table(world, v1, gamma)
	return v0, q0, v1, q1


def build_q_table(world: MiniGridWorld, values: np.ndarray, gamma: float) -> np.ndarray:
	"""根据当前状态价值，计算整张 q-table。

	q_k(s, a) = r(s, a, s') + gamma * v_k(s')
	"""

	q_table = np.zeros((world.n_states, world.n_actions), dtype=float)
	for state in range(world.n_states):
		for action in range(world.n_actions):
			next_state, reward = world.step(state, action)
			q_table[state, action] = reward + gamma * values[next_state]
	return q_table


def value_iteration(
	world: MiniGridWorld,
	gamma: float = 0.9,
	theta: float = 1e-10,
	max_iterations: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
	"""使用值迭代求解最优状态价值。

	更新公式：

		v_{k+1}(s) = max_a [ r(s, a, s') + gamma * v_k(s') ]

	返回：
	- 最终状态价值向量
	- 最终 q-table
	- 每轮迭代的价值历史
	- 每轮迭代的最大变化量
	"""

	values = np.zeros(world.n_states, dtype=float)
	history: List[np.ndarray] = [values.copy()]
	deltas: List[float] = []

	for _ in range(max_iterations):
		q_table = build_q_table(world, values, gamma)
		new_values = np.max(q_table, axis=1)
		delta = float(np.max(np.abs(new_values - values)))

		history.append(new_values.copy())
		deltas.append(delta)
		values = new_values

		if delta < theta:
			break

	final_q_table = build_q_table(world, values, gamma)
	return values, final_q_table, history, deltas


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
	"""从 q-table 中提取贪心策略。

	当多个动作并列最大时，np.argmax 会选择最前面的动作。
	这个规则足够稳定，适合教学示例。
	"""

	return np.argmax(q_table, axis=1)


def format_state_values(values: np.ndarray) -> str:
	"""把一维价值向量格式化成 2x2 网格。"""

	grid = values.reshape(2, 2)
	return np.array2string(np.round(grid, 6), precision=6, suppress_small=True)


def print_q_table(title: str, q_table: np.ndarray) -> None:
	"""在终端打印一个数值 q-table。"""

	print(title)
	header = ["q-value", *ACTION_NAMES]
	print(" ".join(f"{item:^10}" for item in header))
	for state in range(q_table.shape[0]):
		row = [STATE_NAMES[state]]
		for action in range(q_table.shape[1]):
			row.append(_format_cell(float(q_table[state, action])))
		print(" ".join(f"{cell:^10}" for cell in row))


def print_policy_grid(policy: np.ndarray) -> None:
	"""以 2x2 网格打印贪心策略。"""

	print("Greedy policy:")
	for row in range(2):
		cells: List[str] = []
		for col in range(2):
			state = row * 2 + col
			if state == 1:
				cells.append("F")
			elif state == 3:
				cells.append("T")
			else:
				cells.append(ACTION_SYMBOLS[int(policy[state])])
		print(" ".join(f"{cell:^3}" for cell in cells))


def plot_results(
	world: MiniGridWorld,
	q0: np.ndarray,
	q1: np.ndarray,
	v0: np.ndarray,
	v1: np.ndarray,
	filename: str = "value_iteration_q_k0_k1.png",
	show: bool = True,
) -> None:
	"""绘制 k=0 和 k=1 的数值 q-table。"""

	def draw_q_table_panel(ax: plt.Axes, q_table: np.ndarray, title: str, subtitle: str) -> None:
		ax.axis("off")
		ax.set_title(title, fontsize=14, pad=10)

		cell_text = []
		for state in range(world.n_states):
			row = [STATE_NAMES[state]]
			for action in range(world.n_actions):
				row.append(_format_cell(float(q_table[state, action])))
			cell_text.append(row)

		table = ax.table(cellText=cell_text, colLabels=["q-value", *ACTION_NAMES], cellLoc="center", loc="center")
		table.auto_set_font_size(False)
		table.set_fontsize(11)
		table.scale(1.0, 1.55)

		for (row_index, col_index), cell in table.get_celld().items():
			cell.set_edgecolor("#888888")
			cell.set_linewidth(0.8)
			if row_index == 0:
				cell.set_facecolor("#ececec")
				cell.get_text().set_weight("bold")
			elif col_index == 0:
				cell.set_facecolor("#f7f7ff")
				cell.get_text().set_weight("bold")
				cell.get_text().set_color("#1f4e79")

		for state_index in range(world.n_states):
			row_values = q_table[state_index]
			row_max = float(np.max(row_values))
			for action_index in range(world.n_actions):
				if np.isclose(row_values[action_index], row_max):
					cell = table[(state_index + 1, action_index + 1)]
					cell.set_facecolor("#dff0d8")
					cell.get_text().set_weight("bold")

		ax.text(0.5, -0.13, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#555555")

	fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
	fig.suptitle("Direct q-table updates: k=0 and k=1", fontsize=18, weight="bold", y=0.98)
	fig.text(0.5, 0.93, "k=0 uses V0 = [0, 0, 0, 0]; k=1 uses the first updated value vector.", ha="center", va="center", fontsize=11, color="#555555")

	draw_q_table_panel(axes[0], q0, "k=0: q-table computed from V0", f"V0 = {np.array2string(np.round(v0, 1), separator=', ')}")
	draw_q_table_panel(axes[1], q1, "k=1: q-table computed from V1", f"V1 = {np.array2string(np.round(v1, 1), separator=', ')}")

	fig.tight_layout(rect=[0, 0, 1, 0.90])
	fig.savefig(filename, dpi=220, bbox_inches="tight")
	backend = plt.get_backend().lower()
	if show and "agg" not in backend:
		plt.show()
	plt.close(fig)


def main() -> None:
	"""运行一次完整的值迭代演示。"""

	if platform.system() == "Windows":
		sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

	world = MiniGridWorld()
	gamma = 0.9
	v0, q0, v1, q1 = compute_first_two_q_tables(world, gamma)

	print("Value iteration algorithm - direct q-table demo")
	print("k=0 and k=1 are computed explicitly from V0 and V1.")
	print()
	print_q_table("k=0 q-table", q0)
	print()
	print_q_table("k=1 q-table", q1)
	output_path = Path(__file__).resolve().with_name("value_iteration_q_k0_k1.png")
	plot_results(world, q0, q1, v0, v1, filename=str(output_path))
	print(f"Saved figure: {output_path}")

	print()
	print(f"V0 = {np.array2string(np.round(v0, 1), separator=', ')}")
	print(f"V1 = {np.array2string(np.round(v1, 1), separator=', ')}")


if __name__ == "__main__":
	main()
