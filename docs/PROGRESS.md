# 项目进度记录

> 最后更新：2026-07-28 凌晨


## 当前进行中

- **里程碑**：10. 工程化收尾 → **架构升级（ADR-001）**
- **状态**：编排引擎方案已决策，待实施
- **背景**：v1.x 确定性规则生成的谱面质量不可接受——旋律不准、节奏错乱、声部无对话、校验回退失效率高。经系统分析，根因是"将创作任务交给了翻译型规则引擎"，LLM 被锁在门外只做粗粒度回退操作。
- **决策**：详见 `docs/project-charter.md` ADR-001——Pipeline 底层修复（P1）+ LLM 编曲决策（P2）+ LLM 审听微调（P3）。LLM 做编曲师的判断，规则做乐器的执行。
- **三阶段计划**：

| 阶段 | 内容 | 预计改动 |
|------|------|----------|
| **P1：Pipeline 修复** | ① midi_parser 启发式主旋律轨识别；② 输出区分旋律/伴奏；③ tab_generator 接收旋律（不再猜测）+ 保留原始时序 | 4 个文件 |
| **P2：LLM 编曲决策** | 新增 Agent 2.5：歌曲摘要 → LLM → per-section ArrangementPlan（density/bass_style/register/techniques） | 5 个文件 |
| **P3：LLM 审听微调** | 重写校验回退：谱面摘要 → LLM 审听 → 针对性 SectionPlan 调整 → 局部重生成 | 1 个文件（nodes.py） |

- **代码层已完成**（旧方案）：chord voicing、scope override、拍号感知、GM 通道、跳跃密度、空弦泛音、MusicXML 渲染、seed_rag 全量入库
- **今日完成**：
  - Chroma 标签统一：10,550 full_tab → chord_only
  - retriever.py docstring 修正 + PROGRESS.md v2.0 规划
  - Dockerfile 模型预下载 + docker-compose.yml data volume mount
  - 短期记忆（刷新恢复）+ 谱面标题（歌名/文件名）
  - 章程 v1.6→v1.8：CI/CD 交付规范 + v1.0 正式发布里程碑 + ADR-001 架构决策记录
  - 交付清单：`docs/release-checklist.md`
- **待办（旧 M10 剩余）**：Docker 瘦身、CI/CD workflow 编写、单元测试、集成测试、README、LangSmith Trace、v1.0 正式发布
- **待办（新——ADR-001 实施）**：P1→P2→P3 按阶段推进，P1 优先（旋律+节奏是地基）


## 已完成

- [x] **里程碑 1：项目初始化与环境配置**（2026-07-18）
  - `.env` + `src/config.py`（pydantic-settings 统一配置封装）
  - `src/utils/logging.py`（stdlib logging + dictConfig）
  - `src/utils/tracing.py`（LangSmith 默认关闭，M4 开启）
  - `src/api/main.py`（FastAPI + `GET /health` + Swagger UI `/docs`）
  - `Dockerfile`（uv 官方镜像，依赖层缓存优化）
  - `docker-compose.yml`（api + postgres:16 + redis:7 + chromadb:1.5.9，四容器已跑通）
  - ChatDeepSeek 冒烟验证通过（deepseek-v4-pro）
  - **技术发现**：`deepseek-v4-pro` 为 thinking 模型，`max_tokens` 需留足够 reasoning 预算（16 全部被思考消耗 → output 为空；256 正常）。M4 写 Agent 决策节点时注意此点。
- [x] **里程碑 2：数据模型与 State Schema**（2026-07-18）
  - `src/api/schemas.py`：3 个枚举 + 11 个 Pydantic 模型（三层：基础音乐数据 / 聚合输出 / API 交互），中文注释全覆盖
  - `src/agents/state.py`：`AgentState(TypedDict, total=False)` 17 个可选字段 + `create_initial_state()` 工厂，含完整设计决策注释
- [x] **里程碑 3：确定性工具层开发**（2026-07-18）
  - `src/tools/midi_parser.py`：music21 解析 MIDI → 音符序列 + BPM（flatten 拍平全音轨，鼓过滤留到 M5）
  - `src/tools/music21_wrapper.py`：K-S 调性检测 + 半拍分桶和弦识别 + 和弦名标准化（学术名→流行短名）
  - `src/tools/tab_generator.py`：7 层架构（参数区/指板矩阵/zone 硬约束/品位追踪寻址/三声部生成/技巧标注/小节排版），含八度下移低音适配、双策略旋律提取、Travis picking 交替低音
  - `src/tools/tab_validator.py`：4 项物理校验（音域/跨度/横按/空弦合理性）+ 智能变调夹推荐
  - `src/api/schemas.py`：TabGenerationConfig 新增 `melody_source` 字段
  - **技术发现**：music21 10.x 多项 API 变更（`.flat`→`.flatten()`、`.notes` 属性化、`write('midi')` 不生成同时 onset 和弦）均已修正；测试脚本中的和弦失败是 music21 MIDI 写入限制，不影响真实 MIDI 文件
- [x] **里程碑 4：Agent 节点开发**（2026-07-18）
  - `src/agents/nodes.py`：5 Agent 节点 + 入口路由，`caller` 三路分流（首次确定性 / 回退 LLM / 修改确定性执行）
  - `src/agents/graph.py`：StateGraph 6 节点 + 条件边（`should_retry`）+ MIDI/修改双管线入口路由
  - `src/api/schemas.py`：新增 `ModificationOperation`（5 种原子操作）+ `ModificationPlan`
  - `src/tools/tab_generator.py`：新增 `_resolve_scope()` / `_apply_operations()`（三轮强制执行）/ `_normalize_difficulty()`（LLM 容错）/ `_is_strong_beat()`
  - `src/agents/state.py`：新增 `bpm` / `caller` / `modification_plan` 字段
  - 三项 LLM 实测通过（Agent 5 指令解析 / Agent 3 回退修正 / 全图 modify 管线）
  - **架构决策**：5 种原子操作覆盖无限自然语言语义空间；Prompt + 代码双层防御 LLM 枚举值越界
  - **已知限制**：`adjust_difficulty`/`change_density` 当前全局生效（scope override 机制记入 M10 待办）
- [x] **里程碑 5：RAG 曲谱库构建**（2026-07-22）
  - `src/rag/chroma_client.py`：Chroma HttpClient 封装（连接/Collection CRUD/心跳检测）
  - `scripts/seed_rag.py`：预处理流水线（116K MIDI → 过滤 → 特征提取 → 打分 → JSON 报告）
  - `src/rag/indexer.py`：向量化入库（sentence-transformers → Chroma `song_tab_collection`）
  - `src/rag/retriever.py`：检索封装（多语言语义搜索 + `confidence<0.75` 降级兜底规则）
  - LMD 116,189 个 `.mid` 文件已下载并验证（`data/raw_midi/lmd_matched/`）
  - 核心曲库 2 首已入库：押尾コータロー - 黄昏、陈亮 - 无题（`chord_only`, `confidence=1.0`）
  - **技术决策变更**：embedding 模型从 `all-MiniLM-L6-v2`（纯英文）切换为 `paraphrase-multilingual-MiniLM-L12-v2`（中日英多语言）
  - **已知限制**：全量入库推迟到 M10（当前 Chroma 含 7 首测试数据，够 M6-M9 开发使用）


- [x] **里程碑 6：FastAPI 接口开发**（2026-07-23）
  - `src/api/routes.py`：POST /upload（含硬编码输入路由）+ POST /modify + GET /download/{tab_id}（.gp5 导出）
  - `src/api/main.py`：注册 router
  - TabData→guitarpro Song 转换器（层级：Song→Track→Measure→Voice→Beat→Note）
  - 三条路由全部测试通过（上传 MIDI、RAG 搜索、修改、gp5 下载）
  - **架构决策**：难度分级移除——所有谱面统一人手极限约束（`_MAX_FRET=15`，`_MAX_SPAN=4`，空弦不计入跨度）
  - **声部音高阈值**：bass 不接收 >G3(55) 的音，inner 不接收 >C5(72) 的音
  - `Difficulty` 枚举保留为 v2.0 扩展占位，真·指弹难度维度（泛音/AM/加花）记入后续版本规划

- [x] **里程碑 7：前端开发与 alphaTab 集成**（2026-07-23）
  - 静态 HTML 页面（两栏布局：左侧控制面板 380px + 右侧 alphaTab 渲染区，牛皮纸浅黄主题）
  - 上传区域（拖拽 + 点击，支持 .mid / .gp5 / .gpx）+ 歌名搜索框 + 配置面板（风格下拉 + 定弦下拉 5 种常见调弦）
  - 《生成》《修改》按钮 + QA 输入框
  - 自定义播放控制栏（▶ ⏹ + 实时时间；进度条已移除——alphaTab 1.3 playerPosition 只读，无法外部 seek，替代方案为点击六线谱跳转）
  - alphaTab 六线谱渲染（staveProfile: "tab"）+ 本地音色库（TimGM6mb.sf2 5.7MB + sonivox.sf2 1.3MB 备选）
  - .gp5 上传自动编码转换（GBK→UTF-8）+ 下载（UTF-8 编码写出）+ 时值精度优化（三连音/32分音符）+ 滑音方向修复（shiftSlideTo 替代 intoFromBelow）
  - 类型标注修复：midi_parser.py（3 处）、music21_wrapper.py（1 处）、tab_generator.py（2 处）

- [x] **里程碑 8：记忆系统集成**（2026-07-24）
  - `src/memory/checkpointer.py`：PostgresSaver 连接（psycopg 直连 → `builder.compile(checkpointer=...)`）
  - `src/memory/preferences.py`：Redis 用户偏好存储（raw redis-py JSON 键值，不依赖 LangGraph Store 抽象层）
  - `src/agents/graph.py`：编译时注入 Checkpointer（Store 在 API 层用 raw Redis 实现）
  - `src/api/routes.py`：`/upload` + `/modify` 接收 `thread_id`/`user_id`；新增 `GET /preferences` 接口
  - `src/frontend/app.js`：localStorage UUID 管理 + 偏好开关（勾选后下拉框自动填入上次偏好并禁用）
  - **技术决策变更**：LangGraph RedisStore 底层依赖 RediSearch 模块（需 Redis Stack 镜像），我们的 `redis:7-alpine` 不支持 → 改用 raw redis-py 直接存取 JSON 偏好
  - **技术决策变更**：偏好加载从后端覆盖改为前端开关——`load_preferences` 放在 `GET /preferences` 接口由前端主动调用，避免后端覆盖用户手动选择

- [x] **里程碑 9：评估体系（Evals）**（2026-07-24）
  - 层 1 确定性指标：`evals/test_deterministic.py`（pytest, 8/8 通过）——物理校验通过率、旋律保真率、声部 zone 合规率
  - 层 2 LLM 决策层：`evals/eval_llm_nodes.py`（25 条指令，手动触发，需 DEEPSEEK_API_KEY）——Agent 5 指令解析准确率
  - 层 3 RAG 检索：`evals/eval_rag.py`（15 条查询，手动触发）——hit@1 + 正确拒识率
  - 黄金 MIDI 测试集：5 个短 MIDI（~600 字节），存于 `evals/datasets/golden_midi/`
  - **技术决策变更**：章程层 3 原定 hit@3 + 优先级排序 → 精简为 hit@1 + 拒识率（retriever 只取最近邻，M5 的 `_rank()` 已删除）
  - **技术决策变更**：层 2 删除 Agent 3 风格/模板选择一致性（风格是确定性输入参数，非 LLM 决策）
  - **已知限制**：层 3 拒识率 0%（当前 Chroma 仅 7 首，语义空间太稀疏，M10 全量 116K 入库后恢复）


## 已发现并修复的 Bug

- **M5**：midi_parser 零时长音符 → Pydantic 校验崩溃（`duration>0`）→ 加 `dur<=0 → dur=0.001` 防御
- **M5**：Chroma 宿主机连接失败 → `.env` 的 `chromadb:8000`（容器主机名）改为 `localhost:8001`，`docker-compose.yml` 单独覆盖容器内值
- **M5**：sentence-transformers 首次下载需 HuggingFace 网络连接 → 加 `HF_HUB_OFFLINE=1` + `local_files_only=True` 离线模式
- **M5**：indexer 的 `emb_batch` 列表长度始终为 0 → `emb_batch.append()` 位置在 `metas_batch.append({...})` 多行字典之后导致缩进错位，前移修复
- **M5**：`_rank()` 类型优先级破坏语义排序 → 搜"黄昏"返回 type=full_tab 的英文 LMD 文件 → 删除 `_rank()`，信任 Chroma 余弦距离排序
- **M5**：`all-MiniLM-L6-v2` 纯英文模型无法检索中文歌名 → 切换为 `paraphrase-multilingual-MiniLM-L12-v2` 多语言模型
- **M7**：alphaTab 播放无声 → 未配置 soundFont .sf2 音色库，`isReadyForPlayback` 永远 false，`play()` 静默返回 false → 使用 alphaTab 自带轻量 `sonivox.sf2`（1.3MB）+ TimGM6mb.sf2（5.7MB，GPL-2.0）本地文件
- **M7**：alphaTab 播放栏不显示 → 误以为 alphaTab 有内置播放 UI，实际只提供 `play()`/`pause()`/`playPause()` API，需自建 HTML 按钮并绑定 → 自建播放控制栏（▶ ⏹ + 实时时间）
- **M7**：alphaTab beat cursor 不可见 → `.at-cursor-beat` 元素默认 CSS `width: 0`，DOM 存在但肉眼看不见 → 加 CSS `width: 3px` + 红色半透明 + GPU 加速
- **M7**：GP5 中文全部显示为 � → **根因**：GP5 格式底层为单字节编码（CP1252），中文 .gp5 文件的 GBK 字节流被 alphaTab 按 UTF-8 解析 → 非法字节序列 → **修复**：① 后端 Agent 生成的 .gp5：`guitarpro.write(song, buf, encoding="utf-8")` ② 用户上传的 .gp5：后端 `guitarpro.parse(encoding='gbk')` 正确解码后，重写为 UTF-8 版本再返回
- **M7**：alphaTab playerPositionChanged 事件参数猜测错误 → 先猜了 `e.time`、`e.position`、`e` 是数值等，实际应为 `e.currentTime`（毫秒）→ 通过 Console dump 确认 `{ currentTime, endTime, currentTick, endTick, isSeek }`
- **M7**：Agent 生成 .gp5 时值精度不足 → `_ql_to_gp_duration` 仅 8 个离散档位（16分→全音符），缺三连音和 32 分音符 → 扩展至 12 档（新增 32分 + 16分/8分/四分三连音）
- **M7**：滑音方向丢失 → `_technique_to_gp_effect` 对 SLIDE 仅使用 `intoFromBelow`（只上行），下行滑音（5→3）丢失 → 改为 `shiftSlideTo`（alphaTab 根据品差自动判定方向），PULL_OFF 复用同策略
- **M7**：类型标注错误 6 处 → midi_parser.py（`Score | Part | Opus` 类型收窄、`getQuarterBPM()` 不存在）+ music21_wrapper.py（`analyzeStream` 返回 `Key | None`）+ tab_generator.py（lambda 类型收窄、`set[float]` vs `set[int]` 不匹配）→ 全部修复
- **M8**：PostgresSaver.from_conn_string() 在 LangGraph 1.2.9 返回 context manager 而非 saver 对象，`.setup()` 不存在 → 改用 `psycopg.connect()` 直连 + `PostgresSaver(conn)`
- **M8**：LangGraph RedisStore 底层依赖 RediSearch 模块（FT.SEARCH 命令），`redis:7-alpine` 不支持 → 改用 raw redis-py 直接存取 JSON 偏好（`SET/GET fs:prefs:{user_id}`）
- **M8**：偏好覆盖逻辑缺陷——后端 `/upload` 中"默认 jpop 就覆盖为 Redis 偏好"，导致用户手动选 jpop 时也被覆盖 → 改为前端偏好开关：`GET /preferences` 返回偏好 → 前端填开关 → 用户勾选才加载，不覆盖手动选择
- **M10**：guitarpro.py 的 GP5 writer 将多 beat 合并为单 beat → alphaTab 无法正确解析生成的 .gp5（渲染无品位数字），但 Guitar Pro 8 能容错打开 → **前端渲染改为 MusicXML 方案**（`_tabdata_to_musicxml()` 手写 MusicXML，alphaTab 原生支持）; .gp5 下载仍保留 guitarpro.py（GP8/TuxGuitar 能正常打开）
- **M10**：`_build_gp_strings` 弦列表降序排列，guitarpro 的 `Note(string=N)` 引用列表索引而非 `GuitarString.number` → 所有音符错配到相反弦号，GP8 中谱面上下颠倒 → 改 `_build_gp_strings` 按弦号升序排列
- **M10**：MusicXML 中 alphaTab 与 guitarpro 的 staff-tuning line 号约定相反——alphaTab 的 line 1 = 最低音弦 → 六线谱视觉颠倒，音频计算跟着错 → `staff-tuning` line 号改为 `alpha_line = i + 1`（tuning[0]=6弦→line 1），`<string>` 和 `<pitch>` 保持原始弦号不变


## M10 待办优化项（随开发过程积累）

| 优化项 | 说明 | 状态 |
|--------|------|:--:|
| scope override 机制 | `tab_generator.py` 支持 per-measure 参数覆盖表（`overrides` dict），使 `adjust_difficulty` 和 `change_density` 在同一 scope 内写入不同 key（`fret_limit` / `fill_quota`），互不覆盖。完成后删除 `_apply_operations()` Round3 中 `change_density→difficulty` 的临时映射逻辑 | ✅ |
| 拍号感知 scope 解析 | `_apply_operations()` Round1 的 `int(n.start_time / 4.0)` 硬编码了 4/4 拍。改为读取 `harmony.time_signature` 来计算小节边界，确保 3/4、6/8 拍 MIDI 的 scope 解析不偏移 | ✅ |
| 跳跃密度 warning | `tab_validator.py` 新增 `_check_jump_density()`，统计全曲跨弦次数/总音符数，>阈值时发 Warning | ✅ |
| 空弦泛音冲突 warning | `tab_validator.py` 扩展 `_check_open_string_abuse()`，检查低音空弦 pitch class 与和弦根音的音程差，冲突时发 Warning | ✅ |
| Docker 镜像瘦身 | 精简构建阶段、剔除无用依赖、利用 uv 缓存层 | ⬜ |
| seed_rag 多进程加速 + 全量入库 + 距离阈值 | ✅ 多进程加速（4核3workers）+ 全量入库 19,701 首；✅ 距离阈值方向已验证——19K 首下 L2 距离在正确/错误查询间仍重叠（"化学合订本"8.4 < "Yesterday"11.7），当前 multilingual embedding 不适用于此方案，搁置到 v2.0（换模型/两阶段检索） | ✅ |
| 前端渲染方案：GP5 → MusicXML | guitarpro.py 的 GP5 writer 将多 beat 合并为单 beat → alphaTab 无法渲染品位数字 → 前端改为 MusicXML 手写 XML（alphaTab 原生支持），下载仍保留 .gp5 | ✅ |
| midi_parser 保留 GM 通道信息 | 当前 `track=0, channel=0` 硬编码，`guitar_bias` 评分维度永远拿不到真实乐器信息。改为保留 music21 提取的原始 channel/program，使 seed_rag 的吉他偏向评分实际生效。**⚠ 必须在全量入库之前完成** | ✅ |
| chord voicing 同步优化 | `tab_generator.py` 后处理：同时间点 fretted 音符 span 检查，超标时重新分配到邻近弦降低品位跨度。解决《黄昏》剩余 20 个 both_fretted span errors | ✅ |


## alphaTab 1.3 已知限制（非 Bug，受版本/格式所限）

| 限制项 | 说明 | 替代方案 |
|--------|------|----------|
| Beat cursor 闪烁 | alphaTab 1.3 的 cursor 动画机制持续创建/销毁 DOM 元素，`enableAnimatedBeatCursor: false` 会让光标完全消失而非降低帧率 | CSS GPU 加速（`will-change: left`）部分缓解，无法完全消除 |
| playerPosition 只读 | `alphaTabApi.playerPosition = X` 赋值后表面生效（getter 返回新值），但实际播放位置不变——内部 AlphaSynth 引擎不认外部赋值 | 点击六线谱上任意位置即可跳转（alphaTab 内置能力） |
| GP5 不支持 Unicode | GP5 格式为 CP1252 单字节编码，无法原生存储 CJK 文字。当前通过 `encoding='utf-8'` 写入可存储中文，但严格来说不是标准 GP5 | 长期方案：切换到 GPX 格式（UTF-8 原生），需替换 Python 写入库（当前 `guitarpro` 不支持 GPX） |


## 阻塞项

*暂无*


## 当前生效的技术决策

| 领域 | 当前选型 | 确认日期 |
| :--- | :--- | :--- |
| API 框架 | FastAPI（同步路由） | 2026-07-17 |
| Agent 编排 | LangGraph + LangChain | 2026-07-17 |
| LLM | DeepSeek V4 Pro（仅用于决策层） | 2026-07-17 |
| 乐理分析 | music21 | 2026-07-17 |
| 向量数据库 | Chroma（轻量级嵌入式） | 2026-07-17 |
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2`（384 维，中日英多语言） | 2026-07-22 |
| 短期记忆（会话状态） | PostgreSQL（Checkpointer） | 2026-07-17 |
| 长期记忆（用户偏好） | Redis（raw redis-py JSON 键值存取；非 LangGraph Store 抽象层——需 Redis Stack 镜像，当前 `redis:7-alpine` 不支持） | 2026-07-24 |
| 前端渲染 | alphaTab（CDN，静态 HTML） | 2026-07-17 |
| 可观测性 | LangSmith（面试展示 Trace + 决策层评估；开发期默认关闭，M4 起开启） | 2026-07-18 |
| 配置管理 | pydantic-settings（`src/config.py` 统一封装 `.env`，全项目经 `settings.xxx` 取值） | 2026-07-18 |
| 容器化 | Docker Compose | 2026-07-17 |
| CI / 代码质量 | GitHub Actions + Ruff + Pytest（push 仅跑 lint + `tests/unit/` + `evals/`；集成 / E2E 由用户手动执行） | 2026-07-17 |
| CD | GitHub Actions + Docker Hub（tag 触发） | 2026-07-17 |


## 后续版本规划 —— 项目部署生产环境时，所需要优化的内容（搁置到后续版本）

- MP3 / WAV 音频上传扒谱（Demucs + Basic Pitch 链路）
- 用户账号系统（v2.0；跨设备记忆暂不支持，v1.0 以 localStorage UUID 方案代替）
- **多知识库扩充与格式统一**（v2.0）：当前仅 LMD 单一英文源 + 人工标注核心曲库。后续版本接入中文/日文 MIDI 源（如吉他社、Songsterr 等），统一数据预处理管道与元数据格式，提升非英文歌名检索命中率
- **外接网络曲库 API 兜底**（v2.0）：RAG 未命中时，通过第三方乐谱 API（如 Songsterr、Ultimate Guitar 的公开接口）作为回退方案，避免"未找到该曲目"的空白体验
- **真·指弹难度系统**（v2.0）：当前 `Difficulty` 枚举仅控制品位上限（已移除）。真正的指弹难度维度——人工泛音(A.H.)分级标注、击勾弦(H/P)密度控制、AM指法/打板检测、穿插加花密度——全部搁置到后续版本。当前用户通过 QA 修改实现"简化/复杂化"
- **full_tab 难度不匹配提示 + QA 简化**（v2.0）：RAG 命中成品 full_tab 时，若谱面难度与用户设定不符，提示并允许用户通过 QA 降级/升级
- **type 路由启用**（v2.0）：当前 type 标签仅为元数据，所有 RAG 命中均走 Agent 完整链路。待「多知识库扩充与格式统一」和「外接网络曲库 API 兜底」完成后，经典指弹成品谱（full_tab）数量充足且标签可信时，启用 type 路由——full_tab 跳过 Agent 直接渲染，chord_only 走 Agent 完整链路。前置依赖：重新校准 fingerstyle_score 评分配方（当前公式 polyphony+density 两项即达 full_tab 阈值，区分力为零）
- **RAG 检索精度优化**（v2.0）：当前 `paraphrase-multilingual-MiniLM-L12-v2` 模型在歌名检索中无法区分"同一首歌"和"语义相近的不同歌"，拒识率 0%。方案 A：换更强 embedding 模型（BGE-M3 / multilingual-e5-large，区分力更强）；方案 B：两阶段检索——embedding 粗筛 top-20 + cross-encoder 逐对精排，过滤掉语义相近但不同的结果
- 开发过程中产生的新点子、优先级不高的小问题或优化项，由我决定是否放入后续版本


## 关键启动信息（供 Agent 会话恢复时参考）

- 启动时读取：`CLAUDE.md` → `docs/project-charter.md` → `docs/PROGRESS.md`（此处）
- 用户开发模式：VSCode + Claude Code 插件，上下文连续，每天关闭窗口第二天无缝继续
- **⚠️ 7/27 产出物质量攻坚未完成——务必先读 `# 7/27 全天调试总结` 再动手**


---

## 7/27 全天调试总结（7/28 凌晨写，下次会话先读这里）

### 已完成并保留的改动

| 改动 | 文件 | 说明 |
|------|------|------|
| P1 主旋律轨识别（5 维评分+集中度惩罚+note count 权重） | midi_parser.py | 多轨找到正确旋律轨 |
| P1 最低分 40→25 + margin check 移除 | midi_parser.py | 宁选错不混音 |
| Agent 2.5（deepseek-chat V3 + JSON 归一化） | nodes.py/graph.py/schemas.py | LLM 编曲决策 |
| 旋律 timing 保留 | tab_generator.py | 去掉 `start_time = idx * bucket_width` |
| 短期记忆刷新恢复 | app.js | localStorage tabId |
| 谱面标题（歌名/文件名） | routes.py | MusicXML work-title |
| ArrangementPlan JSON 归一化 | nodes.py | "harmonics"→"harmonic" 等模糊匹配 |
| 低音细分拍点 **已回退** | tab_generator.py | 与 ArrangementPlan 组合后 1642 音符爆炸 |

### guitarpro 8h 战争（根本问题仍未解决）

guitarpro.py GP5 writer 三个耦合 bug：Beat 合并、Chord name 255 溢出、ChordAlteration 腐败。
天黑一整天尝试了 15+ 次 duration/beat 数/voice 修改全部失败。死循环：时间聚类→小节跳→0.5s 桶→休止符→固定 duration→无效音长→...

**当前最佳可用态**：旋律 0.5s 桶 + `gp_beat.duration = _ql_to_gp_duration(0.25)`。
无跳音、无无效音长、速度正确。**已知缺陷：每小节末尾有休止符。**

### 根因

不是 guitarpro 的锅——是从 M3 开始三声部混在 TabData 里平铺，导出层无法分离。前端 MusicXML 和下载 .gp5 两条路径各自有各自的 bug。

### 明天方案 A（优先）

guitarpro 双 voice 分离：Voice 0（弦 1-4）+ Voice 1（弦 5-6）。只改 `_tabdata_to_guitarpro_song` 一个函数。

### 今晚最大教训

CLAUDE.md §3 禁止的"猜→改→失败→再猜"循环犯了 ~15 次。第 3 次就应该停下来 diff good/bad .gp5。

### 用户有本地 txt 记录的其他 BUG

（Umbrella 27 小节后静音、前端漏拍/慢、两个格式渲染不一致等——明天从 txt 读取完整清单）
- 遇到不确定内容时主动询问，确认后再执行