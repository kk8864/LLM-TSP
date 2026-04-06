"""
baseline.py —— TSP baseline 对比方法

提供随机路线、最优随机、最近邻贪心，用于和 LLM 结果做横向比较。
"""

import random
from tsp_utils import route_length


def random_route(n, seed=None):
    """返回 0..n-1 的随机排列，当作单次随机路线。"""
    route = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(route)
    return route


def best_random(dist_matrix, n_trials=1000, seed=None):
    """
    重复 n_trials 次随机，返回最短路线。
    用来评估随机搜索的上限，返回 (best_route, best_length)。
    """
    n = len(dist_matrix)
    rng = random.Random(seed)
    best_route = None
    best_len = float("inf")
    for _ in range(n_trials):
        route = list(range(n))
        rng.shuffle(route)
        length = route_length(route, dist_matrix)
        if length < best_len:
            best_len = length
            best_route = route[:]  # 深拷贝，避免被后续修改
    return best_route, best_len


def nearest_neighbor(dist_matrix, start=0):
    """
    最近邻贪心：从 start 出发，每步选最近的未访问城市。
    O(n²) 时间复杂度。
    """
    n = len(dist_matrix)
    visited = [False] * n
    route = [start]
    visited[start] = True
    for _ in range(n - 1):
        current = route[-1]
        # 在所有未访问城市中选距离最近的
        nearest = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: dist_matrix[current][j],
        )
        route.append(nearest)
        visited[nearest] = True
    return route


def best_nearest_neighbor(dist_matrix):
    """枚举所有起点分别跑最近邻，返回其中最优的 (route, length)。"""
    best_route = None
    best_len = float("inf")
    n = len(dist_matrix)
    for start in range(n):
        route = nearest_neighbor(dist_matrix, start=start)
        length = route_length(route, dist_matrix)
        if length < best_len:
            best_len = length
            best_route = route[:]
    return best_route, best_len
