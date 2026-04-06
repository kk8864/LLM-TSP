"""
prompt_builder.py —— OPRO 风格的 Prompt 构造

参考：Large Language Models as Optimizers（Yang et al., 2023）
首轮只描述问题，后续轮将历史路线（最差→最优）注入 Prompt，利用 recency bias 引导改进。
"""


SYSTEM_PROMPT = (
    "You are an expert in combinatorial optimization specializing in the "
    "Traveling Salesman Problem (TSP). "
    "Your goal is to find short Hamiltonian tours by reasoning carefully about "
    "city positions and distances. "
    "You must output exactly one Python list of integer city indices and nothing else."
)


def _format_cities(cities):
    """将城市列表格式化为可读文本块，供 Prompt 使用。"""
    lines = [f"Number of cities: {len(cities)}", "City coordinates (city_id: x, y):"]
    for i, (x, y) in enumerate(cities):
        lines.append(f"  City {i:>2}: ({x:.2f}, {y:.2f})")
    return "\n".join(lines)


def _format_distance_matrix(dist_matrix):
    """将距离矩阵格式化为表格文本，方便 LLM 读取各城市间距离。"""
    n = len(dist_matrix)
    lines = ["Pairwise distances (Euclidean, rounded to 1 decimal):"]
    header = "      " + "  ".join(f"{j:>5}" for j in range(n))
    lines.append(header)
    sep = "      " + "-" * (7 * n)
    lines.append(sep)
    for i in range(n):
        row = f"{i:>3} | " + "  ".join(f"{dist_matrix[i][j]:>5.1f}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)


def _format_history(history, top_k=5):
    """
    从历史记录中取最多 top_k 条结果，按路线长度排序后注入 Prompt。

    排序规则（OPRO 策略）：
      - 最差路线排在最前面
      - 最优路线排在最后（利用 LLM 对最近内容关注度更高的特性）
    """
    if not history:
        return ""

    # 先取 top_k 条最优历史（按长度升序取前 k 条）
    sorted_h = sorted(history, key=lambda x: x["length"], reverse=True)[:top_k]
    # 反转，使最优结果出现在文本末尾（最近位置，LLM 关注度最高）
    sorted_h = list(reversed(sorted_h))

    lines = [
        "",
        f"--- Previous solutions (up to {top_k} shown; best at the bottom) ---",
    ]
    for rank, h in enumerate(sorted_h, 1):
        route_str = ", ".join(str(c) for c in h["route"])
        lines.append(
            f"  #{rank:>2}  length={h['length']:.2f}  route=[{route_str}]"
        )
    lines.append("--- End of previous solutions ---")
    return "\n".join(lines)


def get_system_prompt():
    """返回系统角色 Prompt（固定不变）。"""
    return SYSTEM_PROMPT


def build_prompt(cities, dist_matrix, history=None):
    """
    构造用户侧 Prompt。
    history 为空时只描述问题（首轮），否则注入历史路线让 LLM 改进（OPRO 核心）。
    """
    n = len(cities)
    city_block = _format_cities(cities)
    dist_block = _format_distance_matrix(dist_matrix)
    history_block = _format_history(history) if history else ""

    if not history:
        # 首轮：引导 LLM 从零开始思考，关注城市聚类和避免交叉边
        task_instruction = (
            "Find a short tour that visits every city exactly once and returns "
            "to the starting city.\n"
            "Think step-by-step: consider clusters of nearby cities and avoid "
            "long crossing edges."
        )
    else:
        # 后续轮：告知当前最优值，并引导 LLM 通过分析历史路线进行改进
        best_len = min(h["length"] for h in history)
        task_instruction = (
            f"The best tour found so far has length {best_len:.2f}.\n"
            "Carefully analyse the previous solutions above — identify which "
            "sub-sequences of cities are shared by good solutions and which "
            "transitions are expensive. Then construct a NEW tour that is "
            "strictly shorter than the best solution shown.\n"
            "Strategies to try: 2-opt swaps, relocating expensive cities, "
            "reversing sub-segments."
        )

    # 最终组装完整 Prompt，严格要求 LLM 只输出列表，不输出解释文字
    prompt = f"""\
Solve the following Traveling Salesman Problem (TSP).

{city_block}

{dist_block}
{history_block}

Task:
{task_instruction}

Requirements:
- Visit ALL cities numbered 0 to {n - 1}.
- Each city must appear EXACTLY ONCE in the tour.
- The tour is a closed loop (return to the first city is implicit).

Output format (STRICT):
Output ONLY a single Python list on one line, e.g.:
[0, 4, 2, 7, 1, 5, 3, 9, 8, 6]
Do NOT include any explanation, comments, or extra text — just the list."""

    return prompt
