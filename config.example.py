import os

# ============================================================
# LLM API 配置模板
# 使用方法：将本文件复制为 config.py，填入真实密钥后运行
# 注意：config.py 已在 .gitignore 中，不会被提交到 git
# ============================================================

# API 密钥：优先读取环境变量 LLM_API_KEY，否则使用下方默认值
API_KEY = os.environ.get("LLM_API_KEY", "your-api-key-here")

# API 接入地址，支持所有兼容 OpenAI 格式的服务商：
#   OpenAI  官方:  "https://api.openai.com/v1"
#   DeepSeek:      "https://api.deepseek.com/v1"
#   通义千问 Qwen:  "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   智谱 Zhipu:    "https://open.bigmodel.cn/api/paas/v4"
API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")

# 使用的模型名称，常用选项：
#   "gpt-4o-mini"、"gpt-4o"、"deepseek-chat"、"qwen-plus"、"glm-4-flash"
MODEL_NAME = os.environ.get("LLM_MODEL", "gpt-4o-mini")
