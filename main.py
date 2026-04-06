"""
main.py —— 项目入口

解析命令行参数，依次调用 experiment.run_experiment() 完成实验。

常用命令示例：
  python main.py                             # 默认：10 个城市，10 轮迭代
  python main.py --cities 10 15 20           # 同时运行三种规模
  python main.py --cities 10 --iter 15 --seed 7
  python main.py --cities 10 --model deepseek-chat --api-key sk-...
"""

import argparse
import os
import sys

import config
from experiment import run_experiment


def parse_args():
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="TSP Solver using LLM（OPRO 风格迭代优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cities",
        type=int,
        nargs="+",      # 支持一次传入多个城市规模
        default=[10],
        metavar="N",
        help="城市数量，可指定多个，如 10 15 20（默认：10）",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10,
        dest="n_iter",
        metavar="K",
        help="每次实验的 LLM 迭代轮数（默认：10）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，控制城市坐标生成（默认：42，便于复现）",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results",
        metavar="DIR",
        help="结果保存目录（默认：results/）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="覆盖 config.py 中的模型名称",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="覆盖 config.py 中的 API 密钥",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        metavar="URL",
        help="覆盖 config.py 中的 API 接入地址",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    """将命令行传入的参数覆盖写入 config 模块（运行时动态生效）。"""
    if args.model:
        config.MODEL_NAME = args.model
    if args.api_key:
        config.API_KEY = args.api_key
    if args.api_base:
        config.API_BASE = args.api_base


def check_config():
    """检查 API 密钥是否已配置，未配置时给出提示并退出。"""
    if config.API_KEY in ("", "your-api-key-here"):
        print("=" * 65)
        print("  错误：API 密钥尚未配置！")
        print("  请通过以下任一方式设置：")
        print("    1. 环境变量：  $env:LLM_API_KEY = 'sk-...'")
        print("    2. 编辑文件：  config.py 中 API_KEY = 'sk-...'")
        print("    3. 命令行：    --api-key sk-...")
        print("=" * 65)
        sys.exit(1)


def main():
    args = parse_args()
    apply_cli_overrides(args)   # 先处理命令行覆盖
    check_config()               # 再校验密钥

    # 打印当前运行配置
    print(f"\n  使用模型  : {config.MODEL_NAME}")
    print(f"  API 地址  : {config.API_BASE}")
    print(f"  结果目录  : {os.path.abspath(args.results)}/\n")

    all_results = {}
    # 逐一运行每个城市规模的实验
    for n in args.cities:
        result = run_experiment(
            n_cities=n,
            n_iter=args.n_iter,
            seed=args.seed,
            results_dir=args.results,
        )
        all_results[n] = result

    # 若运行了多个规模，额外打印跨规模汇总表格
    if len(args.cities) > 1:
        print("\n" + "=" * 65)
        print("  多规模实验汇总")
        print("=" * 65)
        header = f"  {'城市数':>6}  {'最近邻长度':>12}  {'LLM最优长度':>12}  {'LLM/最近邻':>10}"
        print(header)
        print("  " + "-" * 48)
        for n, res in all_results.items():
            nn = res["baselines"]["nearest_neighbour"]["length"]
            llm = res["llm"]["best_length"]
            if llm is not None:
                pct = llm / nn * 100
                print(f"  {n:>6}  {nn:>12.2f}  {llm:>12.2f}  {pct:>9.1f}%")
            else:
                print(f"  {n:>6}  {nn:>12.2f}  {'N/A':>12}  {'N/A':>10}")
        print("=" * 65)


if __name__ == "__main__":
    main()
