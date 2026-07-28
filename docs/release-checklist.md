# 交付前待办清单

> 当前处于开发阶段，以下配置为开发便利而设。项目代码全部完成后（M10 末尾）改为交付模式。

---

## docker-compose.yml

| 当前（开发模式） | 交付模式 | 说明 |
|-----------------|---------|------|
| `./src:/app/src:ro` | 删除此挂载 | 代码打入镜像，不需要宿主机源码 |
| `./data:/app/data:ro` | 删除此挂载 | 数据由 `data.tar.gz` 分发，用户解压到项目根目录后手动挂载（或参考 README 的步骤） |

## Dockerfile

| 当前 | 交付模式 | 说明 |
|------|---------|------|
| `RUN uv run python -c "SentenceTransformer(...)"` | 保留 | 模型已在内，交付模式也需要 |

## 待新增

- [ ] `.github/workflows/cd.yml` — tag 触发：云端 build 镜像 → push Docker Hub
- [ ] `scripts/pack_data.sh` — 打包 `data.tar.gz`（min: `data/raw_midi/` + `data/rag/` + `data/curated_fingerstyle/`）
- [ ] README 撰写：
  - 用户使用步骤：下载 Release 附件 → 解压 → 配置 .env → `docker compose up -d`
  - **架构展示**：README 中单独一节引用 `docs/project-charter.md` 的 ADR-001（编排引擎从确定性规则 → LLM 辅助编曲），展示架构权衡与迭代能力

## GitHub Release 手动上传清单（每次打 tag 时）

- [ ] `data.tar.gz`
- [ ] `.env.example`
- [ ] `docker-compose.yml`（交付版，去掉 src/data 挂载）

---

> 此文件在 v1.0 发布后删除。
