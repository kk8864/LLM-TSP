"""
experiment.py —— 实验主控

完整流程：生成城市 → 运行 baseline → LLM 迭代优化 → 打印对比 → 保存结果。
"""

import os
import json
import time

from tsp_utils import (
    generate_cities,
    compute_distance_matrix,
    route_length,
    is_valid_route,
    plot_route,
    plot_convergence,
    plot_comparison,
)
from prompt_builder import build_prompt
from baseline import random_route, best_random, best_nearest_neighbor


# ---------------------------------------------------------------------------
# 日志打印辅助函数
# ---------------------------------------------------------------------------

def _print_separator(char="=", width=65):
    """打印一条分隔线。"""
    print(char * width)


def _print_section(title):
    """打印带标题的分隔块，用于控制台日志的结构化输出。"""
    _print_separator()
    print(f"  {title}")
    _print_separator()


# ---------------------------------------------------------------------------
# 核心实验函数
# ---------------------------------------------------------------------------

def run_experiment(n_cities=10, n_iter=10, seed=42, results_dir="results"):
    """
    运行一次完整的 TSP 实验。

    参数:
        n_cities    -- 城市数量（建议 10 / 15 / 20）
        n_iter      -- LLM 迭代优化轮数
        seed        -- 随机种子，保证城市分布可复现
        results_dir -- 输出目录，保存图片和 JSON 结果

    返回:
        dict，包含所有实验数据（城市、基准结果、LLM 结果、API 统计）
    """
    os.makedirs(results_dir, exist_ok=True)

    _print_section(f"TSP 实验  |  城市数={n_cities}  |  迭代轮数={n_iter}  |  seed={seed}")

    # ------------------------------------------------------------------
    # 步骤 1：生成问题实例
    # ------------------------------------------------------------------
    cities = generate_cities(n_cities, seed=seed)        # 随机生成城市坐标
    dist_matrix = compute_distance_matrix(cities)        # 计算两两距离矩阵

    print(f"\n城市坐标（seed={seed}）：")
    for i, (x, y) in enumerate(cities):
        print(f"  城市 {i:>2}: ({x:.2f}, {y:.2f})")

    # ------------------------------------------------------------------
    # 步骤 2：Baseline 方法
    # ------------------------------------------------------------------
    _print_section("Baseline 方法")

    # 方法一：单次随机路线（最差参考，体现无策略的效果）
    rand_route = random_route(n_cities, seed=seed)
    rand_len = route_length(rand_route, dist_matrix)
    print(f"  随机（单次）      路线: {rand_route}")
    print(f"  随机（单次）      长度: {rand_len:.2f}\n")

    # 方法二：1000 次随机取最优（评估随机搜索的上限）
    t0 = time.time()
    best_rand_route, best_rand_len = best_random(dist_matrix, n_trials=1000, seed=seed)
    print(f"  最优随机（1000次） 路线: {best_rand_route}")
    print(f"  最优随机（1000次） 长度: {best_rand_len:.2f}  ({time.time()-t0:.2f}s)\n")

    # 方法三：最近邻贪心（枚举所有起点取最优，经典启发式）
    t0 = time.time()
    nn_route, nn_len = best_nearest_neighbor(dist_matrix)
    print(f"  最近邻贪心        路线: {nn_route}")
    print(f"  最近邻贪心        长度: {nn_len:.2f}  ({time.time()-t0:.2f}s)")

    # 保存 baseline 路线图
    plot_route(
        cities, rand_route,
        title=f"随机路线  (长度={rand_len:.2f})",
        save_path=os.path.join(results_dir, f"n{n_cities}_random_route.png"),
    )
    plot_route(
        cities, nn_route,
        title=f"最近邻贪心路线  (长度={nn_len:.2f})",
        save_path=os.path.join(results_dir, f"n{n_cities}_nn_route.png"),
    )

    # ------------------------------------------------------------------
    # 步骤 3：LLM OPRO 迭代优化
    # ------------------------------------------------------------------
    _print_section("LLM Solver  （OPRO 风格迭代优化）")

    # 延迟导入，避免在仅运行 baseline 时触发 API 初始化
    from llm_solver import LLMSolver
    solver = LLMSolver()

    history = []          # 存储每轮有效路线：[{"iteration", "route", "length"}, ...]
    best_route = None     # 历史最优路线
    best_len = float("inf")
    start_time = time.time()

    for it in range(1, n_iter + 1):
        print(f"\n  第 {it:>2}/{n_iter} 轮")

        # 首轮不传历史；后续轮将 history 注入 Prompt（OPRO 核心）
        prompt = build_prompt(cities, dist_matrix, history if history else None)

        t0 = time.time()
        route, raw = solver.solve(prompt, n_cities)
        elapsed = time.time() - t0

        # LLM 未返回可解析路线，跳过本轮
        if route is None:
            print(f"    [跳过] LLM 未返回合法路线。")
            continue

        # 双重校验（_parse_route 内部已校验，此处为保险）
        if not is_valid_route(route, n_cities):
            print(f"    [跳过] 路线不合法: {route}")
            continue

        length = route_length(route, dist_matrix)
        history.append({"iteration": it, "route": route, "length": length})

        # 判断是否刷新最优记录
        improved = length < best_len
        if improved:
            best_len = length
            best_route = route[:]  # 深拷贝保存最优路线

        flag = "*** 新最优 ***" if improved else ""
        print(f"    路线 : {route}")
        print(f"    长度 : {length:.2f}   当前最优: {best_len:.2f}   {flag}  ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    llm_stats = solver.stats()

    # ------------------------------------------------------------------
    # 步骤 4：汇总对比表格
    # ------------------------------------------------------------------
    _print_section("实验结果汇总")

    rows = [
        ("随机（单次）",        rand_len,       rand_route),
        ("最优随机（1000次）",  best_rand_len,  best_rand_route),
        ("最近邻贪心",          nn_len,         nn_route),
    ]
    if best_route is not None:
        rows.append(("LLM Solver（最优）", best_len, best_route))

    header = f"  {'方法':<22}  {'路线长度':>10}  {'对比最近邻':>10}"
    print(header)
    print("  " + "-" * 48)
    for name, length, _ in rows:
        pct = length / nn_len * 100
        marker = " <-- 最短" if length == min(r[1] for r in rows) else ""
        print(f"  {name:<22}  {length:>10.2f}  {pct:>9.1f}%{marker}")

    if best_route:
        print(f"\n  LLM 最优路线 : {best_route}")
        print(f"  LLM 最优长度 : {best_len:.2f}")

    print(f"\n  API 调用次数 : {llm_stats['total_calls']}  "
          f"（失败: {llm_stats['failed_calls']}，"
          f"成功率: {llm_stats['success_rate']}）")
    print(f"  总耗时       : {total_time:.1f}s")

    # ------------------------------------------------------------------
    # 步骤 5：保存输出文件
    # ------------------------------------------------------------------

    # LLM 最优路线图
    if best_route is not None:
        plot_route(
            cities, best_route,
            title=f"LLM 最优路线  (长度={best_len:.2f})",
            save_path=os.path.join(results_dir, f"n{n_cities}_llm_best_route.png"),
        )

    # 收敛曲线（叠加 baseline 参考线）
    if history:
        # 只保留两条 baseline 参考线；LLM 最优已由红色实线表示，无需重复画虚线
        baselines_for_plot = {
            "Nearest Neighbour": nn_len,
            "Best Random (1000)": best_rand_len,
        }
        plot_convergence(
            history,
            baselines=baselines_for_plot,
            save_path=os.path.join(results_dir, f"n{n_cities}_convergence.png"),
        )

    # 最近邻 vs LLM 最优 对比图
    if best_route is not None:
        plot_comparison(
            cities,
            {
                "最近邻贪心": (nn_route, nn_len),
                "LLM Solver（最优）": (best_route, best_len),
            },
            save_path=os.path.join(results_dir, f"n{n_cities}_comparison.png"),
        )

    # 将所有结果序列化为 JSON 文件，方便后续查阅和写报告
    results = {
        "n_cities": n_cities,
        "seed": seed,
        "n_iter": n_iter,
        "cities": cities,
        "baselines": {
            "random": {"route": rand_route, "length": rand_len},
            "best_random_1000": {"route": best_rand_route, "length": best_rand_len},
            "nearest_neighbour": {"route": nn_route, "length": nn_len},
        },
        "llm": {
            "best_route": best_route,
            "best_length": best_len if best_route else None,
            "history": history,
            "api_stats": llm_stats,
            "total_time_s": round(total_time, 2),
        },
    }
    json_path = os.path.join(results_dir, f"n{n_cities}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [已保存] {json_path}")

    _print_separator()
    return results
