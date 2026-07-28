/** 指弹吉他谱生成系统 —— 前端交互逻辑
 *
 * 职责：
 *   1. alphaTab 初始化（需 .at-wrap > .at-main 结构，AlphaTabApi 接收 .at-main）
 *   2. 上传区拖拽/点击 → POST /upload → 拿 tab_id → GET /download → alphaTab 渲染
 *   3. 歌名搜索 → 同上流程
 *   4. QA 修改 → POST /modify → 重新下载 .gp5 → alphaTab 刷新渲染
 *   5. 直接上传 .gp5 时，前端本地读取并交给 alphaTab（跳过 Agent 链路）
 *   6. 自定义播放控制栏（alphaTab 不提供内置 UI，播放/暂停/进度/快进全自己写）
 *
 * alphaTab API 参考：https://alphatab.net/docs/api/
 */

(function () {
    "use strict";

    // M8: 首次访问时初始化 UUID（存 localStorage）
    _initUUIDs();

    // =========================================================================
    // DOM 引用
    // =========================================================================
    const $ = (sel) => document.querySelector(sel);

    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    const fileName = $("#file-name");
    const songSearch = $("#song-search");
    const searchBtn = $("#search-btn");
    const styleSelect = $("#style-select");
    const tuningSelect = $("#tuning-select");
    const generateBtn = $("#generate-btn");
    const statusMsg = $("#status-msg");
    const metaInfo = $("#meta-info");
    const metaKey = $("#meta-key");
    const metaTempo = $("#meta-tempo");
    const metaMeasures = $("#meta-measures");
    const metaTuning = $("#meta-tuning");
    const qaSection = $("#qa-section");
    const qaInput = $("#qa-input");
    const modifyBtn = $("#modify-btn");
    const downloadSection = $("#download-section");
    const downloadBtn = $("#download-btn");

    // M8 偏好开关
    const prefToggleRow = $("#pref-toggle-row");
    const prefToggle = $("#pref-toggle");
    const prefSummary = $("#pref-summary");

    // alphaTab 相关
    const atMain = $("#tab-container");     // .at-main 元素，传给 AlphaTabApi
    const placeholder = $("#placeholder");

    // 自定义播放控制栏
    const playerBar = $("#player-bar");
    const btnPlay = $("#btn-play");
    const btnStop = $("#btn-stop");
    const timeLabel = $("#time-label");

    // =========================================================================
    // M8 记忆系统：UUID 管理（localStorage）
    // =========================================================================
    /**
     * user_id   — 长期记忆标识（Redis Store：跨会话恢复偏好）
     * thread_id — 短期记忆标识（PostgresSaver：同会话内刷新恢复状态）
     * 两者均为 UUID，首次访问时生成，存入 localStorage 持久化。
     */
    function _initUUIDs() {
        if (!localStorage.getItem("fs_user_id")) {
            localStorage.setItem("fs_user_id", crypto.randomUUID());
        }
        if (!localStorage.getItem("fs_thread_id")) {
            localStorage.setItem("fs_thread_id", crypto.randomUUID());
        }
    }

    // =========================================================================
    // 状态
    // =========================================================================
    let currentTabId = null;
    let currentTabData = null;
    let alphaTabApi = null;
    let selectedFile = null;
    let totalDuration = 0;       // 谱面总时长（秒）
    /** M8 长期记忆：Redis 中加载的用户偏好 {style, tuning} */
    let savedPrefs = null;

    // =========================================================================
    // M8 偏好加载 + 开关控制
    // =========================================================================
    async function loadSavedPrefs() {
        const uid = localStorage.getItem("fs_user_id");
        if (!uid) return;
        try {
            const resp = await fetch("/preferences?user_id=" + uid);
            if (!resp.ok) return;
            const prefs = await resp.json();
            if (prefs && prefs.style) {
                savedPrefs = prefs;
                prefToggleRow.classList.remove("hidden");
                prefSummary.textContent = "（" + _styleLabel(prefs.style)
                    + " / " + _tuningLabel(_tuningNames(prefs.tuning)) + "）";
                // 默认不勾选，用户主动开才有用
            }
        } catch (e) {
            console.log("[偏好] 加载失败:", e);
        }
    }
    /** 开关切换：禁用/启用下拉框 */
    function onPrefToggle() {
        var on = prefToggle.checked;
        styleSelect.disabled = on;
        tuningSelect.disabled = on;
        if (on && savedPrefs) {
            styleSelect.value = savedPrefs.style;
            tuningSelect.value = savedPrefs.tuning;
        }
    }
    /** 风格枚举值 → 中文名 */
    function _styleLabel(s) {
        return { jpop: "日系", american_folk: "美式", pop_adaptation: "流行" }[s] || s;
    }
    /** 定弦值 → 去掉八度的音符名 */
    function _tuningNames(t) {
        if (!t) return "?";
        return t.split(",").map(function (n) { return n.replace(/[0-9]/g, ""); }).join("");
    }

    // =========================================================================
    // alphaTab 初始化
    // =========================================================================
    function initAlphaTab() {
        if (typeof alphaTab === "undefined") {
            placeholder.textContent =
                "⚠ alphaTab 库加载失败，请检查网络连接\n" +
                "（国内可换用 unpkg：https://unpkg.com/@coderline/alphatab@1.3/dist/alphaTab.min.js）";
            placeholder.style.color = "#c0392b";
            placeholder.style.whiteSpace = "pre-line";
            return;
        }

        console.log("[alphaTab] 库加载成功");

        // alphaTab 要求：new AlphaTabApi( .at-main 元素, settings )
        // 播放链路：soundFont .sf2 音色库 → Web Audio API 合成音频 → 竖线光标走谱
        // 没有 .sf2 → isReadyForPlayback 永远为 false → play() 返回 false 静默失败
        alphaTabApi = new alphaTab.AlphaTabApi(atMain, {
            core: {
                // alphaTab 字体文件路径（CDN 上只有 Bravura.woff，不含 alphaTab.woff）
                fontDirectory: "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.2.3/dist/font/",
            },
            display: {
                layoutMode: "page",        // 页面模式（纵向滚动）
                // 注：GP5 格式底层为单字节编码（CP1252），不支持 CJK 存储。
                // 中文 .gp5 中的 CJK 文字以 GBK 编码存储，alphaTab 按 UTF-8 读取
                // 导致非法字节序列 → 显示为 �。此问题无法通过 canvas 字体配置解决，
                // 根本方案是需要 GPX（UTF-8 原生）格式的 Python 写入库（当前 guitarpro 不支持）。
                // 对于用户上传的中文 .gp5：前端 HTML 层显示元数据，alphaTab 只渲染六线谱。
            },
            player: {
                enablePlayer: true,
                enableCursor: true,
                scrollMode: "vertical",
                // TimGM6mb 音色库（~5.7MB, GPL-2.0），音质优于 alphaTab 自带的 sonivox
                // 备选：sonivox.sf2（~1.3MB，轻量但音质较差）
                soundFont: "TimGM6mb.sf2",
            },
        });
        console.log("[alphaTab] AlphaTabApi 初始化完成");

        // 播放器就绪日志（soundFont 加载完毕后触发）
        alphaTabApi.playerReady.on(() => {
            console.log("[alphaTab] 播放器就绪（soundFont 加载完成），可以播放");
        });

        // 播放器就绪状态变化时输出诊断信息
        alphaTabApi.playerStateChanged.on((e) => {
            const stateNames = { 0: "Paused", 1: "Playing" };
            console.log("[alphaTab] 播放状态: %s", stateNames[e.state] || e.state);
            btnPlay.textContent = e.state === 1 ? "⏸" : "▶";
        });

        // ---- alphaTab 事件 → 播放栏联动 ----

        // 谱子加载完成：获取总时长、初始化进度条
        alphaTabApi.scoreLoaded.on((score) => {
            const trackCount = score.tracks.length;
            const barCount = score.masterBars.length;
            // alphaTab 1.3 的总时长可能在不同路径，逐一尝试
            totalDuration = 0;
            if (score.playbackInfo && score.playbackInfo.duration) {
                totalDuration = score.playbackInfo.duration / 1000;
            }
            console.log("[alphaTab] 谱子加载: %d 音轨, %d 小节, playbackInfo=%s, 总时长=%.1fs",
                trackCount, barCount,
                score.playbackInfo ? "有" : "无",
                totalDuration);
            placeholder.style.display = "none";
            playerBar.classList.remove("hidden");
            updateTimeDisplay(0);
            btnPlay.textContent = "▶";
            console.log("[alphaTab] isReadyForPlayback = %s", alphaTabApi.isReadyForPlayback);
        });

        // 播放位置变化 → 更新进度条和时间
        // alphaTab 1.3 事件结构已确认: { currentTime, endTime, currentTick, endTick, isSeek }
        alphaTabApi.playerPositionChanged.on((e) => {
            // 如果 totalDuration 还没拿到，从事件的 endTime 补上
            // （scoreLoaded 中 score.playbackInfo.duration 可能取不到值）
            if (totalDuration <= 0 && e && typeof e.endTime === "number" && e.endTime > 0) {
                totalDuration = e.endTime / 1000;
                console.log("[alphaTab] 从 playerPositionChanged 获取总时长: %.1f 秒", totalDuration);
            }
            const posMs = (e && typeof e.currentTime === "number") ? e.currentTime : 0;
            const pos = posMs / 1000;
            if (totalDuration > 0) {
                updateTimeDisplay(pos);
            }
        });
    }

    // =========================================================================
    // 播放控制栏交互
    // =========================================================================

    /** 播放 / 暂停切换
     *
     * 关键：alphaTab 需要 soundFont .sf2 加载完毕后才 isReadyForPlayback=true。
     * 在此之前调用 play() 返回 false 且完全静默——不抛异常，不 log，什么都不做。
     */
    function togglePlay() {
        if (!alphaTabApi) return;
        if (!alphaTabApi.isReadyForPlayback) {
            setStatus("⏳ 音色库加载中，请稍后再试...");
            console.log("[播放] isReadyForPlayback 仍为 false，放弃调用 play()");
            return;
        }
        console.log("[播放] 调用 playPause(), 当前状态=%d", alphaTabApi.playerState);
        alphaTabApi.playPause();
    }

    /** 停止 → 回到开头 */
    function stopPlayback() {
        if (!alphaTabApi) return;
        alphaTabApi.stop();
        btnPlay.textContent = "▶";
        updateTimeDisplay(0);
    }

    /** 刷新时间标签：当前 / 总时长 */
    function updateTimeDisplay(currentSec) {
        timeLabel.textContent =
            formatTime(currentSec) + " / " + formatTime(totalDuration);
    }

    /** 秒数 → mm:ss */
    function formatTime(sec) {
        if (!isFinite(sec) || sec < 0) return "00:00";
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
    }

    // =========================================================================
    // 通用 —— 加载谱面 → alphaTab 渲染（优先 MusicXML，回退 .gp5）
    // =========================================================================
    async function loadTabIntoAlphaTab(tabId) {
        // 优先尝试 MusicXML 格式（Agent 管线生成的谱面）
        console.log("[加载] 尝试 /render/%s", tabId);
        let resp = await fetch("/render/" + tabId);
        if (resp.ok) {
            const text = await resp.text();
            console.log("[加载] MusicXML %d chars → alphaTabApi.load()", text.length);
            // alphaTab 1.3 的 load(string) 会把字符串当 URL 去 fetch，
            // 必须转为 Uint8Array 才能正确识别 MusicXML 内容
            alphaTabApi.load(new TextEncoder().encode(text));
            return;
        }
        // 回退 .gp5（用户上传的 .gp5 文件走原路径）
        console.log("[加载] MusicXML 不可用，回退 /download/%s", tabId);
        resp = await fetch("/download/" + tabId);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error("加载谱面失败 (" + resp.status + "): " + errText);
        }
        const buffer = await resp.arrayBuffer();
        console.log("[加载] .gp5 %d bytes → alphaTabApi.load()", buffer.byteLength);
        alphaTabApi.load(new Uint8Array(buffer));
    }

    // =========================================================================
    // 生成流程（.mid / 歌名搜索）
    // =========================================================================
    async function handleGenerate() {
        if (!selectedFile && !songSearch.value.trim()) {
            setStatus("⚠ 请先选择 MIDI 文件或输入歌名");
            return;
        }

        // .gp5 直传：本地读取渲染，不经过 Agent
        if (selectedFile && isGpFile(selectedFile.name)) {
            await handleDirectGp5(selectedFile);
            return;
        }

        // === .mid / 歌名：走 Agent 管线 ===
        generateBtn.disabled = true;
        setStatus("⏳ Agent 管线运行中...");

        try {
            const formData = new FormData();
            if (selectedFile) formData.append("file", selectedFile);
            formData.append("song_name", songSearch.value.trim());
            formData.append("style", styleSelect.value);
            formData.append("tuning", tuningSelect.value);
            // M8: 记忆系统标识
            formData.append("user_id", localStorage.getItem("fs_user_id") || "");
            formData.append("thread_id", localStorage.getItem("fs_thread_id") || "");
            console.log("[生成] style=%s tuning=%s", styleSelect.value, tuningSelect.value);

            const resp = await fetch("/upload", { method: "POST", body: formData });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "上传失败");
            }

            const data = await resp.json();
            currentTabId = data.tab_id;
            currentTabData = data.tab_data;
            console.log("[生成] tab_id=%s measures=%d source=%s",
                data.tab_id,
                data.tab_data ? data.tab_data.measures.length : 0,
                data.source);

            if (data.tab_data) showMeta(data.tab_data);
            setStatus("✅ " + data.message);

            console.log("[生成] 下载 .gp5 → alphaTab 渲染");
            await loadTabIntoAlphaTab(currentTabId);
            console.log("[生成] 完成");

            // 短期记忆：保存 tabId 到 localStorage，刷新后恢复
            localStorage.setItem("fs_last_tab_id", currentTabId);

            showPostGenerate();
        } catch (err) {
            setStatus("❌ " + err.message);
        } finally {
            generateBtn.disabled = false;
        }
    }

    // ---- .gp5 上传：后端 GBK→UTF-8 重编码后，走 /download 取件 → alphaTab 渲染 ----
    async function handleDirectGp5(file) {
        setStatus("⏳ 上传并重编码 Guitar Pro 文件中...");
        try {
            // 1. 上传到后端（触发 GBK→UTF-8 重编码）
            const formData = new FormData();
            formData.append("file", file);
            formData.append("style", styleSelect.value);
            formData.append("tuning", tuningSelect.value);
            formData.append("user_id", localStorage.getItem("fs_user_id") || "");
            formData.append("thread_id", localStorage.getItem("fs_thread_id") || "");
            const resp = await fetch("/upload", { method: "POST", body: formData });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "上传失败");
            }
            const data = await resp.json();
            console.log("[直传] 上传完成, tab_id=%s, message=%s", data.tab_id, data.message);

            if (data.tab_id) {
                currentTabId = data.tab_id;
            }
            currentTabData = data.tab_data;
            setStatus("✅ " + data.message);

            // 2. 下载重编码后的 .gp5 → alphaTab 渲染（中文已正确转码）
            await loadTabIntoAlphaTab(currentTabId);

            if (currentTabId) showPostGenerate();
        } catch (err) {
            setStatus("❌ 加载失败: " + err.message);
        }
    }

    // =========================================================================
    // 修改流程
    // =========================================================================
    async function handleModify() {
        const instruction = qaInput.value.trim();
        if (!instruction) { setStatus("⚠ 请输入修改指令"); return; }
        if (!currentTabId) { setStatus("⚠ 请先生成谱面"); return; }

        modifyBtn.disabled = true;
        setStatus("⏳ Agent 5 解析 → Agent 3 重生成...");

        try {
            const resp = await fetch("/modify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tab_id: currentTabId, instruction }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "修改失败");
            }

            const data = await resp.json();
            currentTabId = data.tab_id;
            currentTabData = data.modified_tab_data;
            setStatus("✅ " + data.changes_summary);

            await loadTabIntoAlphaTab(currentTabId);
            if (currentTabData) showMeta(currentTabData);
            // 短期记忆：保存修改后的 tabId
            localStorage.setItem("fs_last_tab_id", currentTabId);
        } catch (err) {
            setStatus("❌ " + err.message);
        } finally {
            modifyBtn.disabled = false;
            qaInput.value = "";
        }
    }

    // =========================================================================
    // 下载
    // =========================================================================
    function handleDownload() {
        if (!currentTabId) return;
        window.location.href = "/download/" + currentTabId;
    }

    // =========================================================================
    // 辅助
    // =========================================================================
    function isGpFile(name) {
        const lower = name.toLowerCase();
        return lower.endsWith(".gp5") || lower.endsWith(".gpx");
    }
    function setStatus(msg) { statusMsg.textContent = msg; }
    function showPostGenerate() {
        qaSection.classList.remove("hidden");
        downloadSection.classList.remove("hidden");
    }
    function showMeta(tabData) {
        metaInfo.classList.remove("hidden");
        metaKey.textContent = "调性: " + (tabData.key || "?");
        metaTempo.textContent = "BPM: " + (tabData.tempo || "?");
        metaMeasures.textContent = "小节数: " + (tabData.measures ? tabData.measures.length : 0);
        // 定弦显示：从 ["E2","A2","D3","G3","B3","E4"] → "EADGBE (标准)"
        const tuning = tabData.tuning;
        if (tuning && tuning.length === 6) {
            const names = tuning.map(function (s) { return s.replace(/[0-9]/g, ""); }).join("");
            const label = _tuningLabel(names) ? " (" + _tuningLabel(names) + ")" : "";
            metaTuning.textContent = "定弦: " + names + label;
        }
    }
    /** 已知调弦名称映射 */
    function _tuningLabel(names) {
        var map = {
            "EADGBE": "标准",
            "DADGBE": "Drop D",
            "DADGAD": "DADGAD",
            "DGDGBD": "Open G",
            "DADF#AD": "Open D",
        };
        return map[names] || null;
    }

    // =========================================================================
    // 拖拽上传
    // =========================================================================
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) setSelectedFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) setSelectedFile(fileInput.files[0]);
    });
    function setSelectedFile(file) {
        selectedFile = file;
        fileName.textContent = "已选择: " + file.name;
        songSearch.value = "";
    }

    // 歌名搜索 → 清空文件
    songSearch.addEventListener("input", () => {
        if (songSearch.value.trim()) {
            selectedFile = null;
            fileInput.value = "";
            fileName.textContent = "";
        }
    });

    // =========================================================================
    // 按钮 / 键盘事件
    // =========================================================================
    generateBtn.addEventListener("click", handleGenerate);
    searchBtn.addEventListener("click", handleGenerate);
    songSearch.addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleGenerate();
    });
    modifyBtn.addEventListener("click", handleModify);
    qaInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleModify();
    });
    downloadBtn.addEventListener("click", handleDownload);

    // 播放控制栏
    btnPlay.addEventListener("click", togglePlay);
    btnStop.addEventListener("click", stopPlayback);

    // M8 偏好开关
    prefToggle.addEventListener("change", onPrefToggle);

    // =========================================================================
    // 短期记忆：页面刷新后恢复上次谱面
    // =========================================================================
    async function recoverSession() {
        var savedTabId = localStorage.getItem("fs_last_tab_id");
        if (!savedTabId) return;

        console.log("[恢复] 尝试恢复会话 tab_id=%s", savedTabId);
        try {
            var resp = await fetch("/render/" + savedTabId);
            if (!resp.ok) {
                // tab 已过期（服务器重启，内存清空）
                console.log("[恢复] 谱面已过期，清除记录");
                localStorage.removeItem("fs_last_tab_id");
                return;
            }
            // MusicXML 格式加载到 alphaTab
            var text = await resp.text();
            await alphaTabApi.load(new TextEncoder().encode(text), [0]);
            console.log("[恢复] 谱面恢复成功");
            placeholder.style.display = "none";

            // 显示播放栏和下载区
            playerBar.classList.remove("hidden");
            qaSection.classList.remove("hidden");
            downloadSection.classList.remove("hidden");

            // 恢复 tabId 引用
            currentTabId = savedTabId;
        } catch (err) {
            console.log("[恢复] 恢复失败: %s", err.message);
            localStorage.removeItem("fs_last_tab_id");
        }
    }

    // =========================================================================
    // 启动
    // =========================================================================
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            initAlphaTab();
            loadSavedPrefs().then(function () { recoverSession(); });
        });
    } else {
        initAlphaTab();
        loadSavedPrefs().then(function () { recoverSession(); });
    }
})();
