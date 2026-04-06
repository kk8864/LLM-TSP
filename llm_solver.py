"""
llm_solver.py —— LLM 调用封装

支持 OpenAI 格式的 Chat API，解析路线，带重试和统计。
"""

import re
import ast
import time

from openai import OpenAI

import config
from prompt_builder import get_system_prompt
from tsp_utils import is_valid_route


class LLMSolver:
    """大语言模型求解器，封装 API 调用与路线解析逻辑。"""

    def __init__(self):
        # 根据 config.py 中的配置初始化 OpenAI 客户端
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE)
        self.model = config.MODEL_NAME
        # 统计计数器
        self.total_calls = 0   # 成功完成的 API 调用次数
        self.failed_calls = 0  # 最终失败（重试耗尽仍未得到合法路线）的次数

    def solve(self, prompt, n_cities, max_retries=3, temperature=1.0):
        """
        发送 prompt 给 LLM，返回 (route, raw_response)。
        解析失败时自动重试（最多 max_retries 次），全部失败则返回 (None, "")。
        """
        for attempt in range(1, max_retries + 1):
            try:
                # 调用 Chat Completions API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()
                self.total_calls += 1

                # 尝试从回复中提取合法路线
                route = self._parse_route(raw, n_cities)
                if route is not None:
                    return route, raw

                # 解析失败，记录日志后继续重试
                print(f"    [解析失败 {attempt}/{max_retries}] 回复: {raw[:120]}")

            except Exception as exc:
                # API 网络/鉴权/限流等错误，等待后重试
                print(f"    [API 错误 {attempt}/{max_retries}]: {exc}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)  # 退避等待，避免频繁重试

        # 所有重试均失败
        self.failed_calls += 1
        return None, ""

    def _parse_route(self, text, n_cities):
        """
        从 LLM 回复中提取合法 TSP 路线。
        先尝试解析 [...] 列表，失败则提取所有整数取前 n_cities 个。
        """
        # 策略一：在文本中查找所有 [...] 形式的片段，用 ast.literal_eval 解析
        for match in re.findall(r"\[[\d,\s]+\]", text):
            route = self._try_parse_list(match, n_cities)
            if route is not None:
                return route

        # 策略二：提取所有整数，直接取前 n_cities 个作为候选路线
        numbers = re.findall(r"\b\d+\b", text)
        if len(numbers) >= n_cities:
            candidate = [int(x) for x in numbers[:n_cities]]
            if is_valid_route(candidate, n_cities):
                return candidate

        return None

    @staticmethod
    def _try_parse_list(text, n_cities):
        """尝试将字符串解析为合法路线：长度 == n_cities 且包含 0..n-1 各一次。"""
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, list) and len(obj) == n_cities:
                route = [int(x) for x in obj]
                if is_valid_route(route, n_cities):
                    return route
        except Exception:
            pass
        return None

    def stats(self):
        """返回 API 调用统计：总次数、失败次数、成功率。"""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (
                f"{(self.total_calls - self.failed_calls) / max(self.total_calls, 1) * 100:.1f}%"
            ),
        }
