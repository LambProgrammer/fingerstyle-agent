"""FastAPI 应用入口。

启动顺序：加载 .env → 日志 → LangSmith → 创建 app + 路由。
采用同步路由（def），符合 CPU 密集型串行链路的定位。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.utils.logging import setup_logging, get_logger
from src.utils.tracing import configure_tracing

# ---- 启动初始化（模块加载时执行一次）----
setup_logging()
configure_tracing()
logger = get_logger(__name__)

# ---- 创建 FastAPI 实例 ----
app = FastAPI(
    title="指弹吉他谱生成多 Agent 系统",
    description=(
        "MIDI 上传 → 多 Agent 协作 → 指弹谱 (TAB) 生成。"
        "支持难度适配、风格迁移、自然语言局部修改。"
    ),
    version="0.1.0",
)


# ---- 路由 ----
@app.get("/health", tags=["Health"])
def health_check():
    """健康检查：返回服务状态与版本。"""
    return {"status": "ok", "version": app.version}


app.include_router(router)

# 挂载前端静态文件（放在路由注册之后，确保 /health、/docs、/upload 等 API 路由优先匹配）
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

logger.info("FastAPI 应用已创建，Swagger UI: /docs")
