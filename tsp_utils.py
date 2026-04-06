"""
tsp_utils.py —— TSP 基础工具：城市生成、距离矩阵、路线校验与绘图。
"""

import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

# 解决 matplotlib 中文显示乱码/方框问题
# 依次尝试 Windows 常见中文字体，找到即用
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "SimSun", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 修复负号"-"显示异常


def generate_cities(n, seed=None):
    """随机生成 n 个城市坐标，范围 [0, 100]，seed 用于复现。"""
    rng = random.Random(seed)
    return [(round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2)) for _ in range(n)]


def compute_distance_matrix(cities):
    """返回 n×n 欧氏距离矩阵，dist[i][j] 为城市 i 到 j 的直线距离。"""
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                dist[i][j] = math.sqrt(dx * dx + dy * dy)
    return dist


def route_length(route, dist_matrix):
    """计算闭合路线的总长度（最后一个城市自动回到起点）。"""
    n = len(route)
    total = 0.0
    for i in range(n):
        # (i+1) % n 使最后一段自动连回起点，形成闭合环
        total += dist_matrix[route[i]][route[(i + 1) % n]]
    return total


def is_valid_route(route, n):
    """检查路线是否恰好包含 0..n-1 各一次。"""
    if not isinstance(route, list) or len(route) != n:
        return False
    return set(route) == set(range(n))


def plot_route(cities, route, title="TSP Route", save_path=None):
    """
    绘制路线图：红点为城市、蓝线为路线、绿色箭头标注起点方向。
    save_path 非空则保存图片，否则直接弹窗。
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 将路线闭合（末尾加上起点），方便一次性绘制所有边
    route_closed = route + [route[0]]
    tour_xs = [cities[i][0] for i in route_closed]
    tour_ys = [cities[i][1] for i in route_closed]
    ax.plot(tour_xs, tour_ys, "b-", linewidth=1.5, zorder=1, alpha=0.7)

    # 用箭头标注出发方向（起点 → 第二个城市）
    if len(route) >= 2:
        sx, sy = cities[route[0]]
        nx_, ny_ = cities[route[1]]
        ax.annotate(
            "",
            xy=(nx_, ny_),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            zorder=3,
        )

    # 绘制城市节点（红色圆点）
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    ax.scatter(xs, ys, c="red", s=120, zorder=4, edgecolors="black", linewidths=0.8)

    # 在每个城市旁标注编号
    for i, (x, y) in enumerate(cities):
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()


def plot_convergence(history, baselines=None, save_path=None):
    """
    绘制 LLM 迭代收敛曲线。
    baselines 格式：{"方法名": 路线长度, ...}，作为参考水平线叠加。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = [h["iteration"] for h in history]
    lengths = [h["length"] for h in history]

    # 计算历史最优（前缀最小值），用于绘制单调下降的"最优线"
    best_lengths = []
    best = float("inf")
    for l in lengths:
        if l < best:
            best = l
        best_lengths.append(best)

    ax.plot(iterations, lengths, "b-o", label="LLM 每轮结果", alpha=0.55, markersize=5)
    ax.plot(iterations, best_lengths, "r-", linewidth=2.5, label="LLM 历史最优")

    # 叠加 baseline 水平参考线
    if baselines:
        colors = {"Nearest Neighbour": "green", "Best Random (1000)": "orange", "Random": "gray"}
        for name, value in baselines.items():
            color = colors.get(name, "purple")
            ax.axhline(y=value, linestyle="--", linewidth=1.5, color=color, label=name)

    ax.set_xlabel("迭代轮次", fontsize=12)
    ax.set_ylabel("路线总长度", fontsize=12)
    ax.set_title("LLM Solver 收敛曲线 (OPRO-style)", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()


def plot_comparison(cities, routes_dict, save_path=None):
    """
    并排展示多种方法的路线。
    routes_dict 格式：{"方法名": (route, length), ...}，每种方法一个子图。
    """
    n_plots = len(routes_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    # 若只有一个子图，统一包装成列表方便迭代
    if n_plots == 1:
        axes = [axes]

    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]

    for ax, (title, (route, length)) in zip(axes, routes_dict.items()):
        # 绘制该方法的路线
        route_closed = route + [route[0]]
        tx = [cities[i][0] for i in route_closed]
        ty = [cities[i][1] for i in route_closed]
        ax.plot(tx, ty, "b-", linewidth=1.5, alpha=0.7)
        ax.scatter(xs, ys, c="red", s=100, zorder=3, edgecolors="black", linewidths=0.7)
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_title(f"{title}\n路线长度: {length:.2f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"TSP 路线对比（{len(cities)} 个城市）", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [已保存] {save_path}")
    else:
        plt.show()
