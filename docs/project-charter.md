# 指弹吉他谱生成多 Agent 系统 —— 项目章程

> 版本：1.8（架构升级——ADR-001：编排引擎从确定性规则 → LLM 辅助编曲，详见 §4 后 ADR）
> 更新日期：2026-07-26

---

## 1. 场景与核心功能

**场景**：用户上传 MIDI 文件（或通过 RAG 曲谱库搜索歌名），系统通过多 Agent 协作生成可直接演奏的指弹吉他谱（TAB）。系统支持难度适配、风格迁移、基于自然语言的局部修改，并提供谱子播放与下载功能。

**核心功能**：
- MIDI 文件上传 → Agent 完整链路生成指弹谱
- RAG 曲谱库歌名搜索 → 命中后读取 MIDI 文件，经确定性生成链路（解析→和声→指法→校验）输出指弹谱；可通过 QA 修改模式调用 LLM 决策优化
- TAB 谱文件上传（.gp5）→ 直接跳过 Agent 链路，进入前端渲染
- 难度适配（初级 / 中级 / 高级）
- 风格迁移（日系 / 美式 / 流行改编，基于 RAG 检索风格样例）
- 修改模式（用户输入自然语言指令，如"副歌简化"）
- 智能变调夹推荐
- 技巧标注（H / P / B / A.H. 等）
- alphaTab 前端渲染 + 播放
- .gp5 格式下载
- 短期记忆（会话状态恢复）+ 长期记忆（跨会话偏好加载）


## 2. 输入路由（API 层硬逻辑，非 Agent）

**位置**：`src/api/routes.py` 中的 `/upload` 接口内部。

**路由逻辑**（纯硬编码 if-else，不调用 LLM）：

| 用户输入类型 | 检测方式 | 路由目标 | Agent 是否参与 |
|---------|---------|---------|---------------|
| **MIDI 文件（.mid）** | `file.filename.endswith('.mid')` | 走 Agent 完整链路（从 Agent 1 开始） | ✅ 参与 |
| **TAB 谱文件（.gp5）** | `file.filename.endswith('.gp5')` 或 `.gpx` | 直接返回文件路径给前端 alphaTab 渲染 | ❌ 不参与，不走 LangGraph |
| **歌名文本（搜索）** | 请求字段为 `song_name`（非文件） | 走 RAG 检索分支（见下方） | 视命中结果而定 |

**歌名搜索 → RAG 检索分支逻辑**：

1. **RAG 命中**（`type=chord_only` 或 `type=full_tab`）—— 读取命中的 MIDI 文件，经确定性生成链路（解析→和声→指法→校验）输出谱面。之后用户可通过 QA 指令触发 LLM 决策优化
2. **RAG 未命中** —— 提示"未找到该曲目，请上传 MIDI 文件"

> 注：`type` 字段仅作为元数据标记（表示源素材的指弹适配度），不影响路由行为——两种类型均走同样的生成链路。

> 输入路由层不在 LangGraph 中定义，直接写在 FastAPI 路由函数内。


## 3. 技术栈清单

| 组件 | 方案 | 版本 / 约束 | 封装位置 |
|------|------|-------------|----------|
| API 框架 | FastAPI | ≥0.115，使用同步路由（`def`） | `src/api/main.py` / `src/api/routes.py` |
| Agent 编排 | LangGraph + LangChain | ≥0.2 / ≥0.3 | `src/agents/graph.py`, `src/agents/nodes.py` |
| LLM | DeepSeek V4 Pro | API 调用，仅用于决策层 | `src/agents/nodes.py`（Agent 3 / Agent 5） |
| 乐理分析 | music21 | MIT 开源库 | `src/tools/music21_wrapper.py` |
| 向量数据库 | Chroma | 轻量级嵌入式 | `src/rag/chroma_client.py` |
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维，中日英多语言 | `src/rag/indexer.py` |
| 配置管理 | pydantic-settings | ≥2.0 | `src/config.py` |
| Guitar Pro 读写 | guitarpro | ≥0.11 | `src/api/routes.py`（.gp5 导出） |
| 主数据库 | PostgreSQL | ≥15 | `src/memory/checkpointer.py` |
| 缓存 / 长期记忆 | Redis（raw redis-py） | ≥7，AOF+RDB 持久化；非 LangGraph Store 抽象层（需 Redis Stack，当前不采用） | `src/memory/preferences.py` |
| 前端渲染 | alphaTab 1.3 + MusicXML | CDN（JS）+ 本地音色库 + **MusicXML 渲染**（手写 XML，绕开 guitarpro.py GP5 writer bug）；.gp5 下载仍保留（GP8/TuxGuitar 可打开） | `src/frontend/`（index.html + app.js + style.css + *.sf2） + `src/api/routes.py`（`_tabdata_to_musicxml`） |
| 可观测性 / 评估 | LangSmith | 面试展示 Trace + 决策层评估（Datasets / Experiments） | `src/utils/tracing.py`, `evals/eval_llm_nodes.py` |
| 容器化 | Docker Compose | 一键启动 | `docker-compose.yml` |
| CI / 代码质量 | GitHub Actions + Ruff + Pytest | push 时并行运行 lint + 无服务依赖测试（`tests/unit/` + `evals/`，不起外部服务）；集成 / E2E 测试不进 CI，由我手动执行 | `.github/workflows/ci.yml` |
| CD | GitHub Actions + Docker Hub + GitHub Releases | **仅 tag 推送触发**（日常 push 不触发）。CD workflow 在 GitHub 云端构建镜像（不含 `data/`，含 embedding 模型）→ 推送至 Docker Hub；`data.tar.gz` 由本地手动打包上传至 GitHub Release（数据与镜像分离分发，前者变动远少于后者） | `.github/workflows/cd.yml` |


## 4. 系统映射关系（完整）

### 4.1 Agent 节点 → 确定性工具映射

| Agent 节点 | 调用的工具 / 模块 | 输入 | 输出 |
|-----------|------------------|------|------|
| **Agent 1（旋律解析）** | `src/tools/midi_parser.parse_midi()` | MIDI 文件路径或字节流 | 音符序列（`List[Note]`） |
| **Agent 2（和声编排）** | `src/tools/music21_wrapper.analyze_chords()` | 音符序列 | 和弦进行（`List[Chord]`）+ 调性 + BPM |
| **Agent 3（指法生成）** | `src/tools/tab_generator.generate_tab()`（内部调用模板库 `apply_template()` + music21 辅助查询） | 音符序列 + 和弦进行 + 用户配置（难度/风格） | TAB 坐标数据（`List[TabNote]`） |
| **Agent 4（物理校验）** | `src/tools/tab_validator.validate()` | TAB 坐标数据 | 校验结果（通过 / 不通过 + 错误信息） |
| **Agent 5（修改理解器）** | 解析用户指令后，调用 `src/tools/tab_generator.generate_tab()` 重新生成局部 | 自然语言指令 + 当前 TAB 数据 + 修改目标定位 | 修改后的 TAB 数据 |

### 4.2 确定性工具 → 第三方库映射

| 工具模块 | 封装的第三方库 | 核心功能 |
|---------|--------------|----------|
| `src/tools/midi_parser.py` | `music21`（或 `mido`） | 解析 MIDI 文件，提取音符序列、调性、BPM |
| `src/tools/music21_wrapper.py` | `music21` | 和弦识别、调性分析、音符序列处理 |
| `src/tools/tab_generator.py` | **无第三方库（纯 Python 规则 + music21 辅助查询）** | 指法模板库（低音行进 / 填充 / 把位适配）。具体实现细节（如低音行进的具体节奏模式、填充音的选择规则等）在章程中不做穷举，由开发过程中根据实际音乐需求，与我确认后再实现。 |
| `src/tools/tab_validator.py` | 无第三方库（纯 Python 规则） | 物理校验（音域 / 跨度 / 横按合理性） |
| `src/rag/chroma_client.py` | `chromadb` | 向量数据库读写、相似度检索 |
| `src/rag/indexer.py` | `chromadb` + `sentence-transformers` | 曲谱向量化入库 |
| `src/rag/retriever.py` | `chromadb` | 歌名检索、优先级排序 |
| `src/memory/checkpointer.py` | `langgraph.checkpoint.postgres` | PostgreSQL 检查点存储 |
| `src/memory/preferences.py` | `redis`（raw redis-py） | Redis 长期记忆存储（JSON 键值；非 LangGraph Store——依赖 RediSearch 需 Redis Stack 镜像，`redis:7-alpine` 不适用） |

### 4.3 API 路由 → Agent 图映射

| API 路由 | 触发条件 | 执行流程 |
|---------|---------|----------|
| `POST /upload`（含 MIDI 文件） | `file` 是 `.mid` 文件 | 调用 `src/agents/graph.py` 中的 `app.invoke()`，从 Agent 1 开始执行完整链路 |
| `POST /upload`（含 TAB 文件） | `file` 是 `.gp5` / `.gpx` 文件 | **不调用 Agent 图**，后端解析（GBK 编码自动检测）→ UTF-8 重编码 → 存二进制 → 返回 tab_id 供前端下载 |
| `POST /upload`（含歌名文本） | `song_name` 字段有值 | 调用 `src/rag/retriever.retrieve_by_song_name()` → 命中则读取 MIDI 文件，经确定性生成链路（解析→和声→指法→校验）输出谱面；未命中返回"未找到该曲目" |
| `POST /modify` | 用户输入 QA 指令 + 当前已加载谱子 | 调用 `src/agents/graph.py` 中的 `app.invoke()`，从 Agent 5 开始执行局部修改 |
| `GET /render/{tab_id}` | 前端 alphaTab 渲染 | **不调用 Agent 图**，根据 `tab_id` 读取 TabData → 生成 MusicXML → 返回给 alphaTab |
| `GET /download/{tab_id}` | 用户点击下载按钮 | **不调用 Agent 图**，根据 `tab_id` 读取已生成的 TAB 数据，导出为 `.gp5` 文件 |

### 4.4 前端交互 → API 路由映射

| 前端操作 | 调用的 API | 请求内容 |
|---------|-----------|----------|
| 用户选择文件 + 配置面板调整 + 点击《生成》 | `POST /upload` | `file`（MIDI 或 TAB）+ `difficulty` + `style` + `tuning` |
| 用户输入歌名 + 配置面板调整 + 点击《生成》 | `POST /upload` | `song_name` + `difficulty` + `style` + `tuning` |
| 用户在 QA 输入框输入指令 + 点击《修改》 | `POST /modify` | `instruction`（自然语言）+ 当前谱子 ID |
| 用户点击下载按钮 | `GET /download/{tab_id}` | `tab_id`（从当前渲染的谱子获取） |

### 4.5 记忆系统 → LangGraph 集成映射

| 记忆类型 | LangGraph 组件 | 物理存储 | 注入位置 | 作用 |
|---------|---------------|---------|---------|------|
| **短期记忆**（会话状态） | `Checkpointer` | PostgreSQL | 在 `src/agents/graph.py` 中编译 Graph 时传入 `checkpointer=PostgresSaver(conn)` | 保存每次 Agent 调用后的完整 `AgentState`，支持 `thread_id` 恢复 |
| **长期记忆**（用户偏好） | 无 LangGraph 抽象层（raw redis-py） | Redis | 在 `src/api/routes.py` 的 API 层手动调用 `load_preferences`/`save_preferences` | 保存用户跨会话偏好（风格 / 定弦），支持 `user_id` 查询。为何不用 LangGraph RedisStore：底层依赖 RediSearch 模块，需 Redis Stack 镜像（当前 `redis:7-alpine` 不支持） |

> **标识符来源（v1.0 无账号系统）**：前端首次访问时用 `crypto.randomUUID()` 生成 UUID 并存入 localStorage，作为 `user_id`（长期记忆标识）随每次请求携带；`thread_id`（短期记忆会话标识）同样由前端生成并保存在 localStorage，刷新页面后携带原 `thread_id` 恢复会话。换浏览器 / 清缓存即丢失，属"跨设备记忆暂不支持"的既定边界；账号系统搁置到 v2.0。


## 架构决策记录（ADR-001）：编排引擎升级 —— 确定性规则 → LLM 辅助编曲

> **日期**：2026-07-26 | **状态**：P1/P2 已实施，P3 待实施 | **影响范围**：Agent 2.5（新）+ Agent 3 + tab_generator.py + midi_parser.py

### 背景

v1.x 的指法生成引擎（`tab_generator.py`）采用纯确定性规则：旋律提取 = 桶内取最高音；节奏 = 直接复制 MIDI onset；三个声部（低音/内声/旋律）各自独立生成，通过硬 coded zone 约束分配到吉他弦。LLM 仅用于校验失败后的回退（读报错文本 → 输出粗粒度参数调整）和用户 QA 指令解析。

### 问题（实测发现）

1. **旋律不准**：多轨 MIDI 拍平后取 `max(pitch)`，镲片/钢琴加花被误当旋律，人声旋律被淹没
2. **节奏错乱**：多声部分别生成后合并，原始 MIDI 时序在过程中丢失；无节奏量化
3. **声部无对话**：低音、内声、旋律三个生成函数互不知情，输出听感像三个人各弹各的
4. **校验回退失效率高**：LLM 只看到校验报错文本（看不到谱面数据），输出的 5 种粗粒度操作无法定位具体小节/拍位的物理问题，3 次重试经常耗尽后仍不通过
5. **段落无层次**：生成器不知道当前在处理前奏/主歌/副歌，全曲统一参数，音乐上 flat

### 决策

**将 LLM 的角色从"修 bug 的运维脚本"升级为"编曲决策者"**——同时修复 Pipeline 底层缺陷。

**三阶段方案**：

| 阶段 | 内容 | LLM 参与 |
|------|------|:--:|
| **P1：Pipeline 修复** | ① `midi_parser` 新增启发式主旋律轨识别（用复音数/音域/音高方差/GM 乐器加权排序，不依赖轨名）；② 输出区分 `melody_notes` / `accompaniment_notes`；③ `tab_generator` 直接使用传入的旋律音符，保留原始 onset/duration；④ 不再内部调用 `_extract_melody_notes` 猜测旋律 | ❌ 纯规则 |
| **P2：LLM 编曲决策** | 新增 **Agent 2.5**：文本化歌曲摘要 → LLM 输出 per-section `ArrangementPlan`（density / bass_style / register / techniques / dynamic），所有选项来自预定义集合，LLM 做选择题非创作题 | ✅ 1 次分析调用 |
| **P3：LLM 审听微调** | 重写校验回退：不再让 LLM 读报错文本盲猜操作，改为生成**谱面可读摘要**（跳跃统计、跨度分布、段落技巧）→ LLM 基于音乐判断做针对性 SectionPlan 调整 → 局部重生成 | ✅ 1-2 次审听调用 |

**核心原则**：LLM 做编曲师的判断（"副歌这里应该用高把位、加滑音"），规则做乐器的执行（"高把位 = 具体哪根弦哪个品"）。LLM 不创造新规则，只从预定义选项中选择。

### 改动范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/tools/midi_parser.py` | 扩展 | 新增 `_identify_melody_track()` 启发式主旋律识别 |
| `src/api/schemas.py` | 扩展 | 新增 `ArrangementPlan` + `SectionPlan`；`MidiNote` 新增 `is_melody` |
| `src/agents/state.py` | 扩展 | 新增 `melody_notes` + `arrangement_plan` 字段 |
| `src/tools/tab_generator.py` | **重写** | 旋律入口改为接收而非猜测；per-section 参数替代全局统一参数；保留原始时序 |
| `src/agents/nodes.py` | **重写回退逻辑** | 新增 Agent 2.5（LLM 编曲决策）；重写 Agent 3 retry（谱面摘要 → LLM 审听 → SectionPlan 调整） |
| `src/agents/graph.py` | 扩展 | Agent 2 → Agent 2.5（新）→ Agent 3 → Agent 4 |
| `src/api/routes.py` | 微调 | `ArrangementPlan` 传递 |

**不影响**：`music21_wrapper.py`、`tab_validator.py`、RAG 模块、记忆系统、前端。

### 预期效果

- 旋律准确率：从"最高音猜测"提升至"多轨启发式识别 + 作为确定性输入传递给生成器"
- 节奏准确率：从"合并后时序丢失"提升至"保留原始 MIDI onset/duration"
- 音乐层次：从"全曲 uniform"提升至"per-section 差异化编排"
- LLM 调用：从"3 次盲目回退"变为"1 次分析 + 1-2 次有信息的审听"



## 5. 开发里程碑

| 里程碑 | 周期 | 核心产出 | 主要目录 |
|--------|------|----------|----------|
| **1. 项目初始化与环境配置** | 0.5天 | 项目骨架、`docker-compose.yml`（postgres/redis/chromadb）、FastAPI 空壳 + `/health` 接口、Swagger UI 可用、LangSmith 环境变量配置、日志系统（`utils/logging.py`） | 根目录：`pyproject.toml`, `.env`, `docker-compose.yml`, `Dockerfile`；`src/api/main.py`, `src/utils/logging.py` |
| **2. 数据模型与 State Schema** | 1天 | 定义 `AgentState`（LangGraph 状态）、各 Agent 的 I/O Schema（Pydantic 模型） | `src/agents/state.py`, `src/api/schemas.py` |
| **3. 确定性工具层开发（含 MIDI 解析）** | 2-3天 | music21 和弦分析封装、指弹模板库（低音行进 / 旋律填充 / 把位适配）、物理校验器（音域 / 跨度检查）、MIDI 解析工具（单个 `.mid` 文件 → 音符序列） | `src/tools/music21_wrapper.py`, `src/tools/tab_generator.py`, `src/tools/tab_validator.py`, `src/tools/midi_parser.py` |
| **4. Agent 节点开发** | 3-4天 | LangGraph 工作流定义：Agent 1（旋律解析）、Agent 2（和声编排）、Agent 3（指法生成 + LLM 决策）、Agent 4（物理校验 + 回退）、Agent 5（修改理解器）。LangSmith Trace 验证节点流转 | `src/agents/graph.py`, `src/agents/nodes.py` |
| **5. RAG 曲谱库构建** | 2-3天 | Lakh MIDI Dataset 下载、数据预处理（清洗 + 打标签 + 人工标注核心曲库）、向量化入库 Chroma、检索逻辑封装（`retrieve_by_song_name`） | `src/rag/chroma_client.py`, `src/rag/indexer.py`, `src/rag/retriever.py`, `scripts/seed_rag.py`, `data/raw_midi/`, `data/curated_fingerstyle/`, `data/rag/` |
| **6. FastAPI 接口开发** | 1-2天 | 三个接口：`/upload`（含输入路由硬逻辑 + .gp5 GBK→UTF-8 重编码）、`/modify`、`/download`（UTF-8 编码写出 + 时值精度优化 + 滑音方向修复）。TabData→guitarpro Song 转换器 | `src/api/routes.py` |
| **7. 前端开发与 alphaTab 集成** | 1-2天 | 静态 HTML（牛皮纸主题）：拖拽上传 + 歌名搜索 + 风格/定弦下拉配置 + 生成/修改/下载按钮 + QA 输入 + 自定义播放控制栏 + alphaTab 六线谱渲染播放（staveProfile:"tab" + 本地 TimGM6mb.sf2 5.7MB + 备用 sonivox.sf2 1.3MB）+ .gp5 GBK→UTF-8 自动重编码 + 时值精度优化（三连音/32分） + 滑音方向修复 | `src/frontend/`（index.html + app.js + style.css + *.sf2）、`src/api/routes.py`（下载/上传编码修复） |
| **8. 记忆系统集成** | 1-2天 | Checkpointer（PostgreSQL）接入 LangGraph；Store（Redis）接入 LangGraph；在 `graph.py` 编译时注入 | `src/memory/checkpointer.py`, `src/memory/preferences.py`, `src/agents/graph.py` |
| **9. 评估体系（Evals）** | 1-2天 | 三层评估：① 确定性指标（黄金 MIDI 测试集 → 物理校验通过率 / 旋律保真率 / 难度约束符合率，pytest 形式）；② LLM 决策层评估（LangSmith Datasets + `evaluate()`：Agent 5 指令解析准确率、Agent 3 决策一致性）；③ RAG 检索评估（查询集 → hit@1 / hit@3 / 优先级排序正确率） | `evals/`（数据集 + 三层评估代码） |
| **10. 工程化收尾** | 2-3天 | 编写正式测试：单元测试（`tests/unit/`，mock 外部依赖）+ 集成测试（`tests/integration/`，连接真实服务，不进 CI）；GitHub Actions CI（日常 push 触发：并行运行 Ruff + `tests/unit/` + `evals/`，不起外部服务，不构建镜像、不发布 data.tar.gz）；Docker 镜像瘦身（精简构建阶段、剔除无用依赖、利用 uv 缓存层等）；README 撰写；**评估数字采集**——发布前跑三轮评估记录终值：① `uv run pytest evals/test_deterministic.py -v` → 物理校验通过率 / 旋律保真率 / zone 合规率；② `uv run python evals/eval_llm_nodes.py` → Agent 5 指令解析准确率；③ `uv run python evals/eval_rag.py` → hit@1 / 拒识率。数字作为简历性能数据的唯一来源（架构改进不报百分比差值，以设计决策 + Trace 定性展示）；**LangSmith 最终 Trace 收集**——跑一次完整管线，捕获 6 Agent 决策链路可视化 | `tests/unit/`, `tests/integration/`, `.github/workflows/ci.yml`, `README.md`, `evals/` |
| **v1.0 正式发布** | 0.5天 | 代码冻结后执行：① 参照 `docs/release-checklist.md` 检查交付模式配置（docker-compose.yml 去开发挂载、Dockerfile 确认模型内置）；② 手动打包 `data.tar.gz`；③ 手动打 tag 如 `v1.0` → GitHub Actions CD 在云端构建镜像并推送至 Docker Hub；④ 在 GitHub Release 页面手动上传 `data.tar.gz` + `docker-compose.yml`（交付版）+ `.env.example` 作为附件；⑤ 撰写 Release Notes。发布后删除 `docs/release-checklist.md`（一次性检查清单，发布后即失效） | `.github/workflows/cd.yml`, `docs/release-checklist.md` |

### 开发过程中的验证约定（全里程碑通用）

- **开发过程中不为每个里程碑 / 功能单独编写正式单元测试**；正式的单元测试与集成测试统一在里程碑 10 编写并接入 CI。
- 每完成一个功能 / 步骤，由 Agent 用 bash 命令或临时脚本自行验证（导入检查、实例化调用、冒烟运行等），并在进入下一步前向我报告：验证了什么、如何验证、是否通过。
- 我需要手动验证时会主动提出，Agent 需给出基于 Swagger UI（里程碑 1 起可用）的具体测试步骤。
- 例外：评估体系（里程碑 9）的确定性指标测试以 pytest 形式编写于 `evals/test_deterministic.py`，属于该里程碑的正式产出，里程碑 10 时与单元测试一起接入 CI（`uv run pytest tests/unit evals`）。


## 里程碑 5 详细说明：RAG 曲谱库构建与数据预处理

### ⚠️ 启动本里程碑前的准备工作（Agent 职责）

在开始本里程碑的任何代码编写之前，Agent **必须**首先执行以下操作：

1. **输出以下信息，提醒用户手动下载 LMD 数据集**：
   - 官方主页：`http://colinraffel.com/projects/lmd/`
   - 直接下载链接：`http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz`
   - 文件大小：约 1.3GB（压缩包），解压后约 116K 文件
   - 下载后放置位置：`data/raw_midi/` 目录下（需解压）
2. **等待用户确认下载完成**后，再继续后续预处理步骤。
3. **Agent 不得假设文件已存在，也不得自动执行下载**（下载操作由用户在浏览器中手动完成）。

### 数据来源

**主数据源**：Lakh MIDI Dataset（LMD），`lmd_matched` 子集约 11.6 万个 MIDI 文件，压缩包 1.3GB。
**核心曲库（人工标注）**：1-2 首经典指弹曲目（面试演示用），手动收集并强制标记为 `type=chord_only`、`confidence=1.0`（强制走 Agent 链路展示全功能）。后续版本可扩充至 50-100 首。

### 数据预处理流程（共 5 步）

**第 1 步：数据获取与初步筛选**
- 用户手动下载 `lmd_matched.tar.gz`，解压后放入 `data/raw_midi/`
- 遍历所有 `.mid` 文件，做以下过滤：
  - 删除总音符数少于 50 的文件（太短，无意义）
  - 删除时长少于 30 秒的文件（太短，无意义）
  - **不根据音轨数量过滤**（因为指弹 MIDI 通常只有 1 个音轨）
- 产出：`data/raw_midi/` 目录，实扫 116,189 个文件

**第 2 步：MIDI 文件内容分析与打标签**
- 逐一解析 MIDI 文件，提取以下特征：
  - **音符密度**：每小节的音符数平均值
  - **复音数分布**：同时发声的音符数量（指弹通常 2-4 个；伴奏通常 5-10 个）
  - **乐器分配**：音轨使用的 GM 乐器编号
  - **节奏型特征**：音符时值的规律性
- 基于特征计算 `fingerstyle_score`（0-100 分），阈值如下：
  - `≥ 60` → `type = "full_tab"`
  - `30-59` → `type = "chord_only"`
  - `< 30` → 标记为 `"unknown"`，不入库
- 产出：每个 MIDI 文件生成一份分析报告（JSON），包含特征值 + 打分 + 最终标签

**第 3 步：人工标注核心曲库**
- 手动收集 50-100 首经典指弹曲目的 `.mid` 或 `.gp5` 文件
- 放入 `data/curated_fingerstyle/` 目录
- 入库时强制覆盖程序化打标签结果：`type = "chord_only"`，`confidence = 1.0`（强制走 Agent 链路展示全功能）

**第 4 步：向量化与入库（Chroma）**
- 将 MIDI 文件的文本元数据（歌名、艺术家、类型标签、风格标签、调性、BPM）向量化
- Chroma Collection 名称：`song_tab_collection`
- 每个文档包含：
  - `id`：MD5 哈希
  - `metadata`：`{"title": "...", "artist": "...", "type": "full_tab", "style": "...", "confidence": 0.85}`
  - `document`：`"歌名 - 艺术家 - 类型: full_tab - 风格: jpop"`
  - `embedding`：由 sentence-transformers（`paraphrase-multilingual-MiniLM-L12-v2`，384 维，中日英多语言）生成
- 产出：Chroma 持久化数据，存储在 `data/rag/` 目录下

**第 5 步：检索逻辑封装**
- 封装 `retrieve_by_song_name(query)` 函数，供 `/upload` 路由调用
- 检索逻辑：
  1. 在 Chroma 中执行余弦距离向量检索，取 top_k=10
  2. 取最近邻作为最佳命中（信任语义排序，不做 type/confidence 二次重排）
  3. 检索兜底：`confidence < 0.75` 的 `full_tab` 降级为 `chord_only`
  4. 返回最高优先级结果；若无结果则返回 `None`

**验证策略**：
- 先在 **1000 个文件的子集** 上验证打标签规则的准确性
- 全量运行在数据预处理流程中执行


## 里程碑 9 详细说明：评估体系（Evals）

### 目录结构（项目根目录 `evals/`，独立于 `tests/`）

```
evals/
├── datasets/
│   ├── golden_midi/          # 层1 黄金 MIDI 测试集（5-10 个短 MIDI）
│   ├── instructions.jsonl    # 层2 指令解析数据集（上传 LangSmith 的本地源文件）
│   └── rag_queries.jsonl     # 层3 RAG 查询集（~30 条）
├── test_deterministic.py     # 层1 确定性指标（pytest 形式，CI 运行）
├── eval_llm_nodes.py         # 层2 LLM 决策层评估（LangSmith evaluate()，手动触发）
└── eval_rag.py               # 层3 RAG 检索评估（手动触发）
```

> 划分依据：`tests/` 回答"代码对不对"（二值断言），`evals/` 回答"输出质量多好"（指标分数）；评估脚本是反复运行的质量门槛，不属于 `scripts/`（一次性数据准备）的范畴。

### 层 1：确定性指标（无 LLM 成本，随 CI 每次 push 运行）

- **黄金测试集**：5-10 个自制短 MIDI（音阶 / 琶音 / 旋律+和弦），存放于 `evals/datasets/golden_midi/`，体积小、随仓库提交
- **指标（纯规则计算，无 LLM 参与）**：
  - 物理校验通过率：生成 TAB 经 `tab_validator` 校验的通过比例
  - 旋律保真率：TAB 的弦+品反算音高 vs 原 MIDI 旋律音高的覆盖率（music21 确定性计算）
  - 难度约束符合率：初级 → 低把位品位范围、跨度限制等
- 以 pytest 形式实现于 `evals/test_deterministic.py`，里程碑 10 接入 CI（pytest 可从任意指定目录收集测试，目录不必叫 `tests/`）

### 层 2：LLM 决策层评估（LangSmith Datasets + `evaluate()`，手动触发）

- Agent 5 指令解析准确率：20-30 条自然语言指令 → 期望的结构化修改目标（小节范围、操作类型）
- Agent 3 风格 / 模板选择一致性
- 数据集：`evals/datasets/instructions.jsonl`（本地源文件，由脚本上传至 LangSmith Datasets）；脚本：`evals/eval_llm_nodes.py`。涉及 LLM 调用成本，不进 CI，由我决定触发时机

### 层 3：RAG 检索评估（脚本化，手动触发）

- 查询集约 15 条：准确歌名 / 别名 / 错别字 / 中英混合 / 应未命中
- 指标：hit@1（应命中的查询——最近邻是否正确返回）、正确拒识率（不应命中的查询——是否正确返回"未找到"）
- 不评估 hit@3 和优先级排序：当前 retriever 只取最近邻（M5 的 `_rank()` 二次排序已删除，信任 Chroma 余弦距离）
- 查询集文件：`evals/datasets/rag_queries.jsonl`；脚本：`evals/eval_rag.py`

### 范围约束

- **不做**"音乐性 / 好听程度"的 LLM-as-judge 完整流水线（成本高、学习增量低）；主观质量以层 1 客观指标 + 少量人工听感验收为准。


## 6. 关键文件路径

- 环境变量模板：`.env.example`
- FastAPI 入口：`src/api/main.py`
- API 路由定义：`src/api/routes.py`
- 请求/响应 Schema：`src/api/schemas.py`
- LangGraph 图构建：`src/agents/graph.py`
- Agent 节点函数：`src/agents/nodes.py`
- Agent 状态定义：`src/agents/state.py`
- music21 封装：`src/tools/music21_wrapper.py`
- 指法模板库：`src/tools/tab_generator.py`
- 物理校验器：`src/tools/tab_validator.py`
- MIDI 解析工具：`src/tools/midi_parser.py`
- Chroma 客户端：`src/rag/chroma_client.py`
- 曲谱索引器：`src/rag/indexer.py`
- 曲谱检索器：`src/rag/retriever.py`
- 曲谱入库脚本：`scripts/seed_rag.py`
- Checkpointer 配置：`src/memory/checkpointer.py`
- Store 配置：`src/memory/preferences.py`
- 日志配置：`src/utils/logging.py`
- LangSmith 配置：`src/utils/tracing.py`
- 前端静态文件：`src/frontend/index.html`, `src/frontend/app.js`, `src/frontend/style.css`
- 音色库文件：`src/frontend/TimGM6mb.sf2`（5.7MB，默认）, `src/frontend/sonivox.sf2`（1.3MB，备选）
- 评估体系（数据集 + 三层评估）：`evals/`（黄金测试集 `evals/datasets/golden_midi/`、确定性指标 `evals/test_deterministic.py`、LLM 决策层 `evals/eval_llm_nodes.py`、RAG 检索 `evals/eval_rag.py`）
- 单元测试（CI 运行）：`tests/unit/`；集成测试（手动运行）：`tests/integration/`


## 7. 验收标准

**核心功能验收**：
- [ ] 上传 MIDI 文件，Agent 工作流生成指弹谱，alphaTab 渲染并播放
- [ ] 输入歌名，RAG 曲谱库命中 → 读取 MIDI 文件，经确定性生成链路输出指弹谱
- [ ] 输入歌名，RAG 曲谱库未命中 → 提示"未找到该曲目，请上传 MIDI 文件"
- [ ] 命中后可通过 QA 修改模式调用 LLM 决策优化谱面质量
- [ ] 上传 TAB 谱（.gp5），直接跳转 alphaTab 渲染，不走 Agent 链路
- [ ] 配置面板切换难度，生成的指法对应变化（初级→低把位、高级→高把位）
- [ ] 配置面板切换风格，生成的指法风格对应变化
- [ ] QA 输入框输入"副歌简化一点"，系统定位副歌小节并重新生成，其他部分不变
- [ ] 谱面显示技巧标注（H / P / B / A.H. 等）
- [ ] 谱面显示智能变调夹推荐

**记忆系统验收**：
- [ ] 用户刷新页面后，会话状态恢复（短期记忆）
- [ ] 用户关闭页面重开，偏好设置（难度/风格）自动加载（长期记忆）

**评估体系验收**：
- [ ] 确定性指标评估在黄金测试集上运行，产出物理校验通过率 / 旋律保真率 / 难度约束符合率
- [ ] LangSmith 上可见 Agent 5 指令解析评估的 Experiment 结果
- [ ] RAG 检索评估脚本产出 hit@1 / 正确拒识率报告

**工程化验收**：
- [ ] Swagger UI（`/docs`）可交互，所有接口可测试
- [ ] `docker compose up` 一键启动所有服务（含 postgres / redis / chromadb）
- [ ] GitHub Actions CI 在 push 时自动并行运行 Ruff + 无服务依赖测试（`tests/unit/` + `evals/`）
- [ ] 集成测试（`tests/integration/`）与 Docker Compose 端到端测试由我手动执行通过（Agent 在里程碑 10 提醒并提供步骤清单）
- [ ] GitHub Actions CD 仅在 push tag（如 `v1.0`）时触发：云端构建镜像 → Docker Hub；`data.tar.gz` + `docker-compose.yml` + `.env.example` 随 GitHub Release 发布；用户 `docker pull` + 配置 `.env` → `docker compose up -d` 即用
- [ ] LangSmith Trace 记录至少一个完整示例，供面试展示


## 8. 功能边界

### ✅ 当前版本包含
- MIDI 文件上传
- RAG 曲谱库歌名搜索（含 Lakh MIDI Dataset 入库 + 人工标注核心曲库）
- RAG 曲谱库命中后经确定性生成链路输出指弹谱（解析→和声→指法→校验）
- TAB 谱文件上传（直接渲染）
- 难度适配（初级 / 中级 / 高级）
- 风格迁移（日系 / 美式 / 流行改编）
- 修改模式（自然语言指令，局部修改）
- 智能变调夹推荐
- 技巧标注
- alphaTab 渲染 + 播放
- .gp5 下载
- 短期记忆（Checkpointer，PostgreSQL）
- 长期记忆（raw redis-py，JSON 键值存储）
- 工程化（Docker Compose + CI + CD tag-driven + LangSmith）

### ❌ 当前版本不包含 —— 项目部署生产环境时，所需要优化的内容（搁置到后续版本）
- MP3 / WAV 音频上传扒谱（Demucs + Basic Pitch 链路）
- 用户账号系统（v2.0；跨设备记忆暂不支持，v1.0 以 localStorage UUID 方案代替）
- 其他开发过程中产生的新点子、优先级不高的小问题或优化项，由我决定是否放入后续版本


## 9. 后续开发指引

- 本项目章程为技术决策的最终依据。
- 关于开发规范、交互流程和约束规则，请同时阅读根目录下的 `CLAUDE.md`。
- 当前进度请查阅 `docs/PROGRESS.md`。
- 遇到本章程未覆盖的实现细节，优先查阅各技术组件的官方文档，或与我确认后再继续。