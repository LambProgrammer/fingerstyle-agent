# 指弹吉他谱生成多 Agent 系统 —— API 服务镜像
# 基于 uv 官方 Python 3.12 镜像，依赖安装走 uv sync --frozen

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 1. 先只复制依赖申明（最大化 Docker 层缓存：改业务代码不会重新装依赖）
COPY pyproject.toml uv.lock ./

# 2. 安装项目依赖（read-only 模式；此时 src/ 尚未 COPY，uv sync 只关心 pyproject.toml）
RUN uv sync --frozen --no-dev

# 2.5 预下载 embedding 模型到镜像内（避免运行时联网下载；层在 uv.lock 不变时命中缓存）
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 3. 复制业务代码
COPY src/ ./src/

# 4. 暴露端口
EXPOSE 8000

# 5. 启动（不启用 reload，容器中不需要热加载）
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
