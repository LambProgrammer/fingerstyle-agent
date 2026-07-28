"""API 路由定义 —— 三条核心接口 + 输入路由逻辑。

路由职责（对照章程 §2 和 §4.3）：
  POST /upload           输入路由 → 图调用 or RAG 生成 or 直接渲染
  POST /modify           修改管线（Agent 5 → Agent 3 → Agent 4）
  GET /download/{tab_id} .gp5 文件下载

输入路由（纯硬编码 if-else，不调 LLM）：
  .mid 文件          → graph.invoke() 从 Agent 1 起
  .gp5/.gpx 文件     → guitarpro 直接读取，跳过 Agent
  song_name 文本     → RAG 检索 → 命中则读 MIDI → graph.invoke() 从 Agent 1 起
                      未命中 → 404
  无有效输入          → 400

Tab 存储：M6 用内存 dict {tab_id: TabData}，供 modify/download 查找。
重启丢失，当前开发阶段够用。如需持久化，存 PostgreSQL JSON 字段或文件即可，与 Redis 无关。
（Redis Store 是 M8 的长期记忆组件，负责用户偏好/跨会话配置，不存 TabData。）

M8 记忆系统：
  - thread_id → graph.invoke(config={"configurable": {"thread_id": ...}})
    → PostgresSaver 自动存档/恢复完整 AgentState（短期记忆）
  - user_id   → load_preferences / save_preferences
    → Redis Store 持久化风格和定弦偏好（长期记忆）
"""

import io
import tempfile
import uuid
from pathlib import Path
import guitarpro  # type: ignore[import-untyped]
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from src.agents.graph import get_graph
from src.agents.state import create_initial_state
from src.api.schemas import (
    ModifyRequest,
    ModifyResponse,
    TabData,
    TabNote,
    Technique,
    UploadResponse,
)
from src.memory.preferences import load_preferences, save_preferences
from src.rag.retriever import retrieve_by_song_name

router = APIRouter()


# =========================================================================
# GET /preferences —— M8 长期记忆：读取用户偏好
# =========================================================================


@router.get("/preferences", tags=["Preferences"])
def get_preferences(user_id: str = ""):
    """返回用户上次保存的风格和定弦偏好，供前端初始化下拉框和开关。"""
    if not user_id:
        return {}
    return load_preferences(user_id)

# =========================================================================
# Tab 内存存储（M8 迁移到 Redis）
# =========================================================================

_tab_store: dict[str, TabData] = {}
_tab_titles: dict[str, str] = {}  # tab_id → 谱面标题（歌名或上传文件名）
# 用户上传的 .gp5 原始文件存储（key = tab_id, value = 重编码后的 .gp5 二进制）
_gp5_store: dict[str, bytes] = {}


def _store_tab(tab_data: TabData, title: str = "") -> str:
    tab_id = uuid.uuid4().hex[:12]
    _tab_store[tab_id] = tab_data
    if title:
        _tab_titles[tab_id] = title
    return tab_id


def _store_gp5(data: bytes) -> str:
    """存储重编码后的 .gp5 二进制，返回 tab_id 供下载。"""
    tab_id = uuid.uuid4().hex[:12]
    _gp5_store[tab_id] = data
    return tab_id


def _get_tab(tab_id: str) -> TabData | None:
    return _tab_store.get(tab_id)


def _get_gp5(tab_id: str) -> bytes | None:
    return _gp5_store.get(tab_id)


# =========================================================================
# POST /upload —— 核心入口（含输入路由）
# =========================================================================


@router.post("/upload", response_model=UploadResponse, tags=["Upload"])
def upload(
    file: UploadFile | None = File(None),
    song_name: str = Form(""),
    style: str = Form("jpop"),
    tuning: str = Form(""),
    user_id: str = Form(""),       # M8: 长期记忆标识（前端 localStorage UUID）
    thread_id: str = Form(""),     # M8: 短期记忆标识（同会话恢复用）
) -> UploadResponse:
    """统一上传入口 —— 硬编码路由分发。

    M8: 若提供 user_id，优先读取 Redis 偏好补全空缺参数；
    生成成功后回写偏好；若提供 thread_id，图从上次 checkpoint 恢复。
    """
    # 路由 1: 文件上传
    if file and file.filename:
        filename = file.filename.lower()
        if filename.endswith((".gp5", ".gpx")):
            return _handle_gp_file(file, filename)
        if filename.endswith((".mid", ".midi")):
            return _handle_midi_file(file, style, tuning, thread_id, user_id)
        raise HTTPException(400, f"不支持的文件格式: {file.filename}")

    # 路由 2: 歌名搜索
    if song_name.strip():
        return _handle_song_search(song_name.strip(), style, tuning, thread_id, user_id)

    # 路由 3: 无输入
    raise HTTPException(400, "请上传文件或输入歌名")


def _handle_gp_file(file: UploadFile, filename: str) -> UploadResponse:
    """处理 .gp5/.gpx：GBK 解析 → UTF-8 重编码 → 存二进制 → 返回 tab_id。

    中文 Guitar Pro 5 用 GBK 编码存文字，alphaTab 按 UTF-8 读 → 乱码。
    解决方案：后端重编码为 UTF-8 后返回，前端走 /download/{tab_id} 取件。
    """
    content = file.file.read()

    # 编码检测：中文 .gp5 常用 GBK，英文用 CP1252。GBK 是 ASCII 超集，
    # 对纯英文文件无副作用，优先尝试 GBK。
    encodings = ["gbk", "cp1252"]
    song = None
    for enc in encodings:
        try:
            song = guitarpro.parse(io.BytesIO(content), encoding=enc)
            break
        except Exception:
            continue

    if song is None:
        raise HTTPException(400, f"Guitar Pro 文件解析失败，尝试了编码: {encodings}")

    # 重编码为 UTF-8 后写出
    buf = io.BytesIO()
    guitarpro.write(song, buf, encoding="utf-8")
    buf.seek(0)
    gp5_data = buf.read()
    tab_id = _store_gp5(gp5_data)

    # 提取元数据用于前端展示
    artist_info = f" - {song.artist}" if song.artist else ""
    return UploadResponse(
        tab_id=tab_id, tab_data=None, source="direct_gp5",
        message=f"'{song.title or filename}'{artist_info} 已加载",
    )


def _handle_midi_file(
    file: UploadFile,
    style: str,
    tuning: str,
    thread_id: str = "",
    user_id: str = "",
) -> UploadResponse:
    """处理 .mid → graph.invoke() 从 Agent 1 起。

    M8: thread_id 传入 config 激活 PostgresSaver 短期记忆；
         user_id 用于回写偏好到 Redis Store 长期记忆。
    """
    midi_path = _save_upload(file)
    parsed_tuning = _parse_tuning(tuning)
    tab_data = _run_graph(midi_path, style, parsed_tuning, thread_id=thread_id)
    # 取文件名（不含扩展名）作为谱面标题
    assert file.filename is not None  # 入口处已检查 endswith('.mid')
    title = Path(file.filename).stem
    tab_id = _store_tab(tab_data, title=title)
    # 长期记忆：生成成功后保存偏好
    if user_id:
        save_preferences(user_id, style, tuning)
    return UploadResponse(
        tab_id=tab_id, tab_data=tab_data, source="midi_pipeline",
        message=f"MIDI 文件 '{file.filename}' 已生成指弹谱",
    )


def _handle_song_search(
    query: str,
    style: str,
    tuning: str,
    thread_id: str = "",
    user_id: str = "",
) -> UploadResponse:
    """歌名搜索 → RAG 检索 → 读命中 MIDI → graph.invoke()。"""
    hit = retrieve_by_song_name(query)
    if hit is None:
        raise HTTPException(404, f"未找到曲目 '{query}'，请上传 MIDI 文件")
    midi_path = hit["file_path"].replace("\\", "/")
    if not Path(midi_path).exists():
        raise HTTPException(500, f"RAG 命中的文件不存在: {midi_path}")
    parsed_tuning = _parse_tuning(tuning)
    tab_data = _run_graph(midi_path, style, parsed_tuning, thread_id=thread_id)
    tab_id = _store_tab(tab_data, title=hit["title"])
    if user_id:
        save_preferences(user_id, style, tuning)
    artist_info = f" - {hit['artist']}" if hit.get("artist") else ""
    return UploadResponse(
        tab_id=tab_id, tab_data=tab_data, source="midi_pipeline",
        message=f"RAG 命中: {hit['title']}{artist_info} (type={hit['type']})",
    )


# =========================================================================
# POST /modify —— QA 修改
# =========================================================================


@router.post("/modify", response_model=ModifyResponse, tags=["Modify"])
def modify(req: ModifyRequest) -> ModifyResponse:
    """QA 修改 → Agent 5（LLM 解析）→ Agent 3（确定性执行）→ Agent 4（校验）。"""
    current_tab = _get_tab(req.tab_id)
    if current_tab is None:
        raise HTTPException(404, f"谱面不存在: {req.tab_id}")

    state = create_initial_state(
        difficulty="beginner", style=current_tab.style,
        tuning=current_tab.tuning,
    )
    state["modify_instruction"] = req.instruction
    state["tab_data"] = current_tab.model_dump()

    result = get_graph().invoke(state)
    if result.get("error"):
        raise HTTPException(500, f"修改失败: {result['error']}")
    modified_raw = result.get("tab_data", {})
    if not modified_raw:
        raise HTTPException(500, "修改后未生成新的谱面数据")

    modified_tab = TabData.model_validate(modified_raw)
    _tab_store[req.tab_id] = modified_tab

    # 提取修改摘要 + 校验结果（即使校验未全通过也返回当前最佳结果）
    plan = result.get("modification_plan", {})
    summary = plan.get("summary", "") if plan else "修改完成"
    validation = result.get("validation", {})
    if not validation.get("is_valid", True):
        err_count = len(validation.get("errors", []))
        summary += f"（校验存在 {err_count} 个物理约束问题，已尽力修正）"

    return ModifyResponse(
        tab_id=req.tab_id, modified_tab_data=modified_tab, changes_summary=summary,
    )


# =========================================================================
# GET /download/{tab_id} —— .gp5 文件下载
# =========================================================================


@router.get("/render/{tab_id}", tags=["Render"])
def render_tab(tab_id: str):
    """返回 MusicXML 格式——供 alphaTab 前端渲染六线谱。

    guitarpro.py 的 GP5 writer 产出的 .gp5 对 alphaTab 不兼容（beat 被合并），
    改为 MusicXML（W3C 开放标准，alphaTab 原生支持，music21 原生导出）。
    """
    tab_data = _get_tab(tab_id)
    if tab_data is None:
        raise HTTPException(404, f"谱面不存在: {tab_id}")

    musicxml_str = _tabdata_to_musicxml(tab_data, title=_tab_titles.get(tab_id, ""))
    return Response(
        content=musicxml_str,
        media_type="application/vnd.recordare.musicxml+xml",
    )


@router.get("/download/{tab_id}", tags=["Download"])
def download(tab_id: str):
    """导出谱面文件。

    Agent 生成的谱面 → .gp5（GP8/TuxGuitar 打开）；
    用户上传的 .gp5 → 重编码后的 .gp5 原样返回。
    """
    # 来源 1: Agent 管线生成的 TabData → .gp5
    tab_data = _get_tab(tab_id)
    if tab_data is not None:
        try:
            gp_song = _tabdata_to_guitarpro_song(tab_data)
            buf = io.BytesIO()
            guitarpro.write(gp_song, buf, encoding="utf-8")
            buf.seek(0)
            return Response(
                content=buf.getvalue(),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{tab_id[:8]}.gp5"',
                },
            )
        except Exception:
            raise HTTPException(
                500,
                "guitarpro.py 无法处理该谱面的 .gp5 导出（已知第三方库限制）。"
                "请尝试其他歌曲，或使用前端的播放/下载功能。",
            )

    # 来源 2: 用户上传 .gp5 → 重编码后的二进制
    gp5_data = _get_gp5(tab_id)
    if gp5_data is not None:
        return Response(
            content=gp5_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{tab_id[:8]}.gp5"',
            },
        )

    raise HTTPException(404, f"谱面不存在: {tab_id}")


# =========================================================================
# 辅助函数
# =========================================================================


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.mid").suffix or ".mid"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        return tmp.name


def _parse_tuning(tuning_str: str) -> list[str]:
    if not tuning_str.strip():
        return ["E2", "A2", "D3", "G3", "B3", "E4"]
    parts = [t.strip() for t in tuning_str.split(",") if t.strip()]
    return parts if len(parts) == 6 else ["E2", "A2", "D3", "G3", "B3", "E4"]


def _run_graph(
    midi_path: str,
    style: str,
    tuning: list[str],
    thread_id: str = "",
) -> TabData:
    """执行 Agent 管线，返回 TabData。Checkpointer 注入后 config 为必传项。"""
    state = create_initial_state(midi_path=midi_path, style=style, tuning=tuning)
    # Checkpointer 注入后 graph.invoke() 必须提供 thread_id。
    # 前端未传时用临时 UUID（本次调用不走 checkpoint 恢复，但仍需满足 API 要求）
    invoke_config: dict = {
        "configurable": {"thread_id": thread_id or uuid.uuid4().hex}
    }
    try:
        result = get_graph().invoke(state, config=invoke_config)  # type: ignore[arg-type]
    except Exception as exc:
        raise HTTPException(500, f"Agent 管线执行失败: {exc}")
    if result.get("error"):
        raise HTTPException(500, f"生成失败: {result['error']}")
    tab_raw = result.get("tab_data", {})
    if not tab_raw:
        raise HTTPException(500, "Agent 管线未生成谱面数据")
    return TabData.model_validate(tab_raw)


# =========================================================================
# TabData → GuitarPro Song 转换器
# =========================================================================


def _tabdata_to_guitarpro_song(tab: TabData) -> guitarpro.models.Song:
    """将 TabData 转换为 guitarpro Song 对象，用于导出 .gp5。

    guitarpro 层级：Song → Track → Measure(header) → Voice → Beat → Note

    方案 B 双 voice 分离：
      Voice 0（弦 1-3）：旋律 + 内声部（voice="melody"|"inner"|""）
      Voice 1（弦 4-6）：低音线（voice="bass"）
    每个 voice 独立建桶、独立设 beat，绕过 guitarpro.py 单 voice 过载导致的
    beat 合并 / chord name 溢出 / ChordAlteration 腐败三个耦合 bug。
    """
    from guitarpro import models as gm

    gp_song = gm.Song()
    gp_song.tempo = tab.tempo
    track = gp_song.tracks[0]
    track.name = "Fingerstyle Guitar"
    track.strings = _build_gp_strings(tab.tuning)
    track.settings.notation = False  # .gp5 下载时只渲染六线谱
    track.measures.clear()

    # 声部优先级：去重用（高值优先保留）
    _VOICE_PRIORITY = {"melody": 3, "inner": 2, "bass": 1, "": 0}
    # 小节时长（quarter lengths）：后续累加计算每小节在曲中的绝对起始位置
    _measure_dur = tab.measures[0].time_signature[0] if tab.measures else 4
    _song_time = 0.0  # 累加器：当前小节在曲中的绝对起始 quarter length

    for m in tab.measures:
        header = gm.MeasureHeader()
        gp_measure = gm.Measure(track=track, header=header)

        # 去重：同一弦同一时刻只保留最高优先级声部的音符
        deduped_notes = _dedup_same_string(m.notes, _VOICE_PRIORITY)

        # 按 voice 字段分流（空字符串 → Voice 0，兼容旧数据）
        voice0_notes = [n for n in deduped_notes if n.voice in ("melody", "inner", "")]
        voice1_notes = [n for n in deduped_notes if n.voice == "bass"]

        # 当前小节的实际时长（支持变拍号）
        md = float(m.time_signature[0])

        _fill_gp_voice(gp_measure.voices[0], voice0_notes, md, _song_time)
        _fill_gp_voice(gp_measure.voices[1], voice1_notes, md, _song_time)

        track.measures.append(gp_measure)
        _song_time += md

    return gp_song


def _dedup_same_string(notes: list[TabNote], priority: dict[str, int]) -> list[TabNote]:
    """同一弦同一时刻冲突时，保留优先级最高的音符（melody > inner > bass）。

    吉他每根弦同时只能弹一个音——此函数修复 tab_generator 内声部与旋律
    在同弦冲突的边界情况，同时避免触发 guitarpro.py 多 voice 写入 bug。
    """
    groups: dict[tuple[float, int], list[TabNote]] = {}
    for n in notes:
        key = (n.start_time, n.string)
        groups.setdefault(key, []).append(n)

    result: list[TabNote] = []
    for key, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            best = max(group, key=lambda n: priority.get(n.voice, 0))
            result.append(best)
    return result


def _fill_gp_voice(voice, notes: list[TabNote], measure_duration: float = 4.0,
                   measure_start_ql: float = 0.0) -> None:
    """将一组 TabNote 填入一个 guitarpro Voice。

    beat duration 使用间隔时长：每个 beat 的 duration = 到下一个 beat 的时间差。
    最后一个 beat 的 duration = 到小节末尾的时间差。
    这样 beat 之间没有空隙，无需休止符填充。
    """
    from guitarpro import models as gm

    if not notes:
        return

    buckets: dict[float, list[TabNote]] = {}
    for tn in notes:
        buckets.setdefault(tn.start_time, []).append(tn)

    sorted_times = sorted(buckets)
    for idx, start_time in enumerate(sorted_times):
        notes_at_t = buckets[start_time]
        rel_start = start_time - measure_start_ql  # 小节内相对位置（quarter lengths）
        gp_beat = gm.Beat(voice=voice)
        gp_beat.start = int(rel_start * 960)

        # 间隔时长：填满到下一个 beat 或小节末尾
        if idx + 1 < len(sorted_times):
            gap_ql = sorted_times[idx + 1] - start_time
        else:
            gap_ql = measure_duration - rel_start

        gp_beat.duration = _ql_to_gp_duration(gap_ql)

        for tn in notes_at_t:
            gp_note = gm.Note(beat=gp_beat, value=tn.fret, string=tn.string)
            gp_note.effect = _technique_to_gp_effect(tn.technique)
            gp_beat.notes.append(gp_note)

        voice.beats.append(gp_beat)


def _tabdata_to_musicxml(tab: TabData, title: str = "") -> str:
    """TabData → MusicXML 六线谱（手写 XML，不使用 music21）。

    直接构建 MusicXML 字符串，每个 <note> 包含 <string> 和 <fret>
    标签，alphaTab 渲染为六线谱品位数字。
    """
    # 计算每小节起始时间
    divisions = 960  # ticks per quarter note

    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"')
    parts.append('  "http://www.musicxml.org/dtds/partwise.dtd">')
    parts.append('<score-partwise version="4.0">')
    parts.append(f'  <work><work-title>{title or "Fingerstyle Tab"}</work-title></work>')
    parts.append('  <identification><creator type="composer">Fingerstyle Agent</creator></identification>')

    # --- part-list ---
    parts.append('  <part-list>')
    parts.append(_build_score_part_xml(tab.tuning))
    parts.append('  </part-list>')

    # --- part ---
    parts.append('  <part id="P1">')

    for m_idx, m in enumerate(tab.measures):
        # 第一小节放完整属性
        if m_idx == 0:
            parts.append(f'    <measure number="{m.number}" implicit="no">')
            parts.append('      <attributes>')
            parts.append(f'        <divisions>{divisions}</divisions>')
            parts.append(f'        <time><beats>{m.time_signature[0]}</beats>')
            parts.append(f'        <beat-type>{m.time_signature[1]}</beat-type></time>')
            parts.append('        <clef><sign>TAB</sign><line>5</line></clef>')
            parts.append('        <staff-details>')
            parts.append('          <staff-lines>6</staff-lines>')
            for i, name in enumerate(tab.tuning):
                import music21.note as m21note  # type: ignore[import-untyped]
                pitch = m21note.Note(name).pitch.midi
                st = pitch % 12
                step_map = {
                    0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D#", 1),
                    4: ("E", 0), 5: ("F", 0), 6: ("F#", 1), 7: ("G", 0),
                    8: ("G#", 1), 9: ("A", 0), 10: ("A#", 1), 11: ("B", 0),
                }
                step, alter = step_map[st]
                alpha_line = i + 1  # line 1 = tuning[0]=E2（6弦）
                parts.append(f'          <staff-tuning line="{alpha_line}">')
                parts.append(f'            <tuning-step>{step}</tuning-step>')
                parts.append(f'            <tuning-alter>{alter}</tuning-alter>')
                parts.append(f'            <tuning-octave>{pitch // 12 - 1}</tuning-octave>')
                parts.append('          </staff-tuning>')
            parts.append('          <capo>0</capo>')
            parts.append('        </staff-details>')
            parts.append('      </attributes>')
            # 速度
            parts.append('      <direction placement="above">')
            parts.append('        <direction-type>')
            parts.append('          <metronome><beat-unit>quarter</beat-unit>')
            parts.append(f'          <per-minute>{tab.tempo}</per-minute></metronome>')
            parts.append('        </direction-type>')
            parts.append(f'        <sound tempo="{tab.tempo}"/>')
            parts.append('      </direction>')
        else:
            parts.append(f'    <measure number="{m.number}" implicit="no">')

        # 音符：按 start_time 排序，同时刻的加 <chord/>
        sorted_notes = sorted(m.notes, key=lambda x: x.start_time)
        prev_time = None
        for tn in sorted_notes:
            is_chord = (tn.start_time == prev_time)
            prev_time = tn.start_time

            dur_ticks = int(tn.duration * divisions)
            if dur_ticks <= 0:
                dur_ticks = divisions // 2

            pitch = _reverse_pitch(tn.string, tn.fret, tab.tuning)
            st = pitch % 12
            step_map = {
                0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D#", 1),
                4: ("E", 0), 5: ("F", 0), 6: ("F#", 1), 7: ("G", 0),
                8: ("G#", 1), 9: ("A", 0), 10: ("A#", 1), 11: ("B", 0),
            }
            step, alter = step_map[st]
            octave = pitch // 12 - 1

            parts.append('      <note>')
            if is_chord:
                parts.append('        <chord/>')
            parts.append('        <pitch>')
            parts.append(f'          <step>{step}</step>')
            if alter != 0:
                parts.append(f'          <alter>{alter}</alter>')
            parts.append(f'          <octave>{octave}</octave>')
            parts.append('        </pitch>')
            parts.append(f'        <duration>{dur_ticks}</duration>')
            parts.append('        <type>eighth</type>')
            parts.append('        <notations>')
            parts.append('          <technical>')
            parts.append(f'            <string>{tn.string}</string>')
            parts.append(f'            <fret>{tn.fret}</fret>')
            parts.append('          </technical>')
            parts.append('        </notations>')
            parts.append('      </note>')

        parts.append('    </measure>')

    parts.append('  </part>')
    parts.append('</score-partwise>')
    return "\n".join(parts)


def _build_score_part_xml(tuning: list[str]) -> str:
    """生成 MusicXML <score-part> 含 6 弦吉他 tab staff 定义。"""
    import music21.note as m21note  # type: ignore[import-untyped]
    # MIDI semitone → (step_name, alter)
    _SEMITONE_MAP = [
        (0, "C"), (1, "C"), (2, "D"), (3, "D#"), (4, "E"),
        (5, "F"), (6, "F#"), (7, "G"), (8, "G#"), (9, "A"),
        (10, "A#"), (11, "B"),
    ]
    _ALTER_MAP = [
        0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
    ]
    lines = []
    for i, name in enumerate(tuning):
        string_num = 6 - i   # tuning[0]=E2=弦6
        pitch = m21note.Note(name).pitch.midi
        st = pitch % 12
        step = _SEMITONE_MAP[st][1]
        alter_val = _ALTER_MAP[st]
        octave = pitch // 12 - 1
        lines.append(
            f'        <staff-tuning line="{string_num}">'
            f'<tuning-step>{step}</tuning-step>'
            f'<tuning-alter>{alter_val}</tuning-alter>'
            f'<tuning-octave>{octave}</tuning-octave>'
            f'</staff-tuning>'
        )

    return (
        '<score-part id="P1">\n'
        '  <part-name>Guitar Tab</part-name>\n'
        '  <score-instrument id="P1-I1">\n'
        '    <instrument-name>Acoustic Guitar</instrument-name>\n'
        '  </score-instrument>\n'
        '  <midi-instrument id="P1-I1">\n'
        '    <midi-channel>1</midi-channel>\n'
        '    <midi-program>25</midi-program>\n'
        '  </midi-instrument>\n'
        '  <staff-details>\n'
        '    <staff-lines>6</staff-lines>\n'
        + "\n".join(lines) + "\n"
        '    <capo>0</capo>\n'
        '  </staff-details>\n'
        '</score-part>\n    '
    )


def _reverse_pitch(string: int, fret: int, tuning: list[str]) -> int:
    """弦+品 → MIDI 音高。"""
    import music21.note as m21note  # type: ignore[import-untyped]
    open_pitches = {6 - i: m21note.Note(name).pitch.midi for i, name in enumerate(tuning)}
    return open_pitches.get(string, 40) + fret


def _build_gp_strings(tuning: list[str]) -> list[guitarpro.models.GuitarString]:
    """音名列表 → guitarpro GuitarString 列表（升序，number=1 在索引 0）。"""
    import music21.note as m21note  # type: ignore[import-untyped]
    strings = []
    for i, name in enumerate(tuning):
        string_num = 6 - i  # tuning[0]=E2=弦6
        pitch = m21note.Note(name).pitch.midi
        strings.append((string_num, pitch))
    # 按弦号升序排列（guitarpro Note.string 引用列表索引）
    strings.sort(key=lambda x: x[0])
    return [guitarpro.models.GuitarString(number=num, value=pitch) for num, pitch in strings]


def _ql_to_gp_duration(ql: float) -> guitarpro.models.Duration:
    """quarterLength → guitarpro Duration。

    采用更细粒度的时值映射，减少精度损失。
    支持：32分音符、16分/8分三连音、附点变体。
    """
    gm = guitarpro.models
    T = gm.Tuplet
    if ql <= 0.125:
        return gm.Duration(value=32)                                # 32nd
    if ql <= 0.167:
        return gm.Duration(value=16, tuplet=T(enters=3, times=2))    # 16th triplet
    if ql <= 0.25:
        return gm.Duration(value=16)                                # 16th
    if ql <= 0.333:
        return gm.Duration(value=8, tuplet=T(enters=3, times=2))     # 8th triplet
    if ql <= 0.5:
        return gm.Duration(value=8)                                 # 8th
    if ql <= 0.667:
        return gm.Duration(value=4, tuplet=T(enters=3, times=2))     # quarter triplet
    if ql <= 0.75:
        return gm.Duration(value=8, isDotted=True)                  # dotted 8th
    if ql <= 1.0:
        return gm.Duration(value=4)                                 # quarter
    if ql <= 1.5:
        return gm.Duration(value=4, isDotted=True)                  # dotted quarter
    if ql <= 2.0:
        return gm.Duration(value=2)                                 # half
    if ql <= 3.0:
        return gm.Duration(value=2, isDotted=True)                  # dotted half
    return gm.Duration(value=1)                                     # whole


def _technique_to_gp_effect(technique: Technique) -> guitarpro.models.NoteEffect:
    """TabNote 技巧标注 → guitarpro NoteEffect。

    guitarpro NoteEffect 支持的技巧：hammer、slides（列表）、vibrato、palmMute 等。
    pullOff 在 guitarpro 中无直接对应属性，使用 shiftSlideTo 模拟下行滑音。
    shiftSlideTo 根据前后品差自动判定滑音方向，支持 3→5（上行）和 5→3（下行）。
    """
    effect = guitarpro.models.NoteEffect()
    if technique == Technique.HAMMER_ON:
        effect.hammer = True
    elif technique in (Technique.PULL_OFF, Technique.SLIDE):
        # 滑音统一使用 shiftSlideTo：alphaTab 根据品差自动识别方向
        effect.slides = [guitarpro.models.SlideType.shiftSlideTo]
    return effect
