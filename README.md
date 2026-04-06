# 基于大语言模型的 TSP 求解器（OPRO 风格）

> 参考论文：*Large Language Models as Optimizers*（Yang et al., 2023）

本项目将 LLM（大语言模型）作为黑盒优化器，通过迭代 Prompt 的方式求解小规模 TSP（旅行商问题），支持 10 / 15 / 20 个城市的实例，并与随机方法、最近邻贪心方法进行对比。

---

## 项目结构

```
TSP/
├── main.py            # 程序入口，解析命令行参数并启动实验
├── experiment.py      # 实验主控：OPRO 迭代循环 + 结果对比 + 文件保存
├── llm_solver.py      # LLM API 封装：调用接口、解析路线、重试机制
├── prompt_builder.py  # OPRO 风格 Prompt 构造：首轮描述 + 后续注入历史
├── tsp_utils.py       # 基础工具：城市生成、距离矩阵、路线校验、绘图
├── baseline.py        # Baseline 方法：随机路线、最优随机、最近邻贪心
├── config.py          # 配置文件：API 密钥、模型名称、接入地址
├── requirements.txt   # Python 依赖包列表
└── results/           # 自动创建，存储实验输出的图片和 JSON 数据
```

---

## 环境准备

### 第一步：安装依赖

```powershell
pip install -r requirements.txt
```

依赖说明：
- `openai >= 1.0.0`：用于调用兼容 OpenAI 格式的 Chat API
- `matplotlib >= 3.5.0`：用于绘制路线图和收敛曲线

### 第二步：配置 LLM API

**方式一：设置环境变量（推荐，密钥不写入代码）**

```powershell
# Windows PowerShell
$env:LLM_API_KEY  = "sk-..."                              # 必填
$env:LLM_MODEL    = "deepseek-chat"                       # 可选，默认 gpt-4o-mini
$env:LLM_API_BASE = "https://api.deepseek.com/v1"        # 可选，默认 OpenAI 官方
```

**方式二：直接编辑 `config.py`**

```python
API_KEY    = "sk-..."
API_BASE   = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
```

**支持的服务商（均兼容 OpenAI 格式）：**

| 服务商 | `API_BASE` | `MODEL_NAME` 示例 |
|--------|-----------|------------------|
| OpenAI 官方 | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 智谱 Zhipu | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |

---

## 运行方法

```powershell
# 最简运行：10 个城市，10 轮迭代
python main.py

# 同时运行三种规模（写报告推荐）
python main.py --cities 10 15 20 --iter 10

# 自定义轮数和随机种子
python main.py --cities 10 --iter 15 --seed 7

# 命令行直接指定模型和密钥（无需改 config.py）
python main.py --cities 10 --model deepseek-chat --api-key sk-...
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cities` | `10` | 城市数量，可指定多个，空格分隔，如 `10 15 20` |
| `--iter` | `10` | LLM 迭代优化轮数 |
| `--seed` | `42` | 随机种子，控制城市坐标生成 |
| `--results` | `results/` | 结果输出目录 |
| `--model` | *(config.py)* | 覆盖模型名称 |
| `--api-key` | *(config.py)* | 覆盖 API 密钥 |
| `--api-base` | *(config.py)* | 覆盖 API 接入地址 |

---

## 输出文件说明

实验结束后，`results/` 目录下会生成以下文件（`N` 为城市数量）：

| 文件名 | 内容 |
|--------|------|
| `n{N}_random_route.png` | 随机路线图 |
| `n{N}_nn_route.png` | 最近邻贪心路线图 |
| `n{N}_llm_best_route.png` | LLM 最优路线图 |
| `n{N}_convergence.png` | LLM 迭代收敛曲线（含 baseline 参考线） |
| `n{N}_comparison.png` | 最近邻 vs LLM 最优路线并排对比图 |
| `n{N}_results.json` | 完整数值结果（每轮路线、长度、API 统计等） |

---

## 算法原理（OPRO）

本项目实现的核心思路来自 OPRO 论文中的 **meta-prompt 优化策略**：

1. **初始化**：随机生成城市坐标，计算欧氏距离矩阵。
2. **第 1 轮**：仅向 LLM 描述问题（城市坐标 + 距离矩阵），要求其输出初始路线。
3. **第 k 轮（k > 1）**：将历史中最多 5 条路线按"最差→最优"顺序注入 Prompt——最优路线排在末尾，利用 LLM 的近端注意力偏好（recency bias）使其更关注最佳方案，进而生成更短的新路线。
4. **路线校验**：每轮返回的路线须通过合法性验证（包含所有城市各一次），不合法则跳过。
5. **结果对比**：将 LLM 最优结果与单次随机、最优随机（1000次）、最近邻贪心三种 baseline 进行量化比较。
