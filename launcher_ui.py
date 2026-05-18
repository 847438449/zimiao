"""Modern Gradio launcher for the YOLOv8 realtime inference pipeline.

This dashboard keeps config.ini as the single source of truth and manages the
run.py subprocess lifecycle from a lightweight UI.
"""

from __future__ import annotations

import configparser
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"

DASHBOARD_CSS = """
#title {text-align: center; margin-bottom: 0.25rem;}
#subtitle {text-align: center; color: #64748b; margin-bottom: 1.25rem;}
.status-box textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
.live-log textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.35;}
"""

HARDWARE_LABEL = "📹 实时硬件采集 (Hardware Mode)"
VIDEO_LABEL = "🎬 离线视频仿真 (Video Mode)"
IMAGE_LABEL = "🖼️ 静态图片盲测 (Image Mode)"
SOURCE_LABELS = [HARDWARE_LABEL, VIDEO_LABEL, IMAGE_LABEL]
MODE_BY_LABEL = {
    HARDWARE_LABEL: "hardware",
    VIDEO_LABEL: "video",
    IMAGE_LABEL: "image",
}
LABEL_BY_MODE = {mode: label for label, mode in MODE_BY_LABEL.items()}
DEFAULT_SOURCE_PATH = r"F:\yolo_training\game_test..mp4"
RES_1080_TO_1080 = "1080P 仿真源 -> 1080P 物理屏 (1.0x)"
RES_1080_TO_2K = "1080P 仿真源 -> 2K 物理屏 (1.33x)"
RES_2K_TO_2K = "2K 真实硬件 -> 2K 物理屏 (1.0x)"
RES_MODE_CHOICES = [RES_1080_TO_1080, RES_1080_TO_2K, RES_2K_TO_2K]
RES_MATRIX_BY_MODE = {
    RES_1080_TO_1080: {"scale": 1.0, "crop_width": 320, "crop_height": 320},
    RES_1080_TO_2K: {"scale": 1.3333, "crop_width": 320, "crop_height": 320},
    RES_2K_TO_2K: {"scale": 1.0, "crop_width": 448, "crop_height": 448},
}
RES_MODE_ALIAS = {
    "1080P -> 1080P": RES_1080_TO_1080,
    "1080P -> 2K": RES_1080_TO_2K,
    "2K -> 2K": RES_2K_TO_2K,
    "2K 仿真源 -> 2K 物理屏 (1.0x)": RES_2K_TO_2K,
}

DEBUG_OPTIONS = {
    "显示渲染窗口 (show_window)": "show_window",
    "显示目标边界框 (show_boxes)": "show_boxes",
    "显示类别标签 (show_labels)": "show_labels",
    "显示置信度数值 (show_conf)": "show_conf",
}

_process_lock = threading.Lock()
_pipeline_process: subprocess.Popen | None = None
_pipeline_log_file = None


def _new_parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve existing mixed-case option names
    parser.read(CONFIG_PATH, encoding="utf-8")
    return parser


def _ensure_schema(config: configparser.ConfigParser) -> None:
    if not config.has_section("Detection window"):
        config.add_section("Detection window")
    if not config.has_section("Capture Methods"):
        config.add_section("Capture Methods")
    if not config.has_section("Aim"):
        config.add_section("Aim")
    if not config.has_section("Mouse"):
        config.add_section("Mouse")
    if not config.has_section("AI"):
        config.add_section("AI")
    if not config.has_section("Debug window"):
        config.add_section("Debug window")
    if not config.has_section("Control_Filter"):
        config.add_section("Control_Filter")

    config.setdefault("Detection window", {})
    config.setdefault("Capture Methods", {})
    config.setdefault("Aim", {})
    config.setdefault("Mouse", {})
    config.setdefault("AI", {})
    config.setdefault("Debug window", {})
    config.setdefault("Control_Filter", {})

    if not config.has_option("Detection window", "detection_window_width"):
        config.set("Detection window", "detection_window_width", "320")
    if not config.has_option("Detection window", "detection_window_height"):
        config.set("Detection window", "detection_window_height", "320")
    if not config.has_option("Detection window", "circle_capture"):
        config.set("Detection window", "circle_capture", "True")
    if not config.has_option("Capture Methods", "simulation_mode"):
        config.set("Capture Methods", "simulation_mode", "True")
    if not config.has_option("Capture Methods", "source_mode"):
        legacy_sim = config.getboolean("Capture Methods", "simulation_mode", fallback=True)
        config.set("Capture Methods", "source_mode", "video" if legacy_sim else "hardware")
    if not config.has_option("Capture Methods", "source_path"):
        config.set(
            "Capture Methods",
            "source_path",
            config.get("Capture Methods", "simulation_video_path", fallback=DEFAULT_SOURCE_PATH),
        )
    if not config.has_option("Capture Methods", "simulation_video_path"):
        config.set("Capture Methods", "simulation_video_path", DEFAULT_SOURCE_PATH)
    if not config.has_option("Aim", "head_shot_ratio"):
        config.set("Aim", "head_shot_ratio", "0.3")
    if not config.has_option("Mouse", "mouse_sensitivity"):
        config.set("Mouse", "mouse_sensitivity", "3.0")
    if not config.has_option("AI", "AI_conf"):
        config.set("AI", "AI_conf", "0.2")
    for option in DEBUG_OPTIONS.values():
        if not config.has_option("Debug window", option):
            config.set("Debug window", option, "True")
    if not config.has_option("Control_Filter", "cooperative_filtering"):
        config.set("Control_Filter", "cooperative_filtering", "True")
    if not config.has_option("Control_Filter", "tag_color_density_threshold"):
        config.set("Control_Filter", "tag_color_density_threshold", "0.10")
    if not config.has_option("Control_Filter", "resolution_scale_factor"):
        config.set("Control_Filter", "resolution_scale_factor", "1.3333")
    if not config.has_option("Control_Filter", "current_res_mode"):
        config.set("Control_Filter", "current_res_mode", "1080P -> 2K")


def normalize_res_mode(config_value: str) -> str:
    value = (config_value or "").strip().strip('"').strip("'")
    if value in RES_MODE_CHOICES:
        return value
    if value in RES_MODE_ALIAS:
        return RES_MODE_ALIAS[value]
    if "1080P" in value and "2K" in value:
        return RES_1080_TO_2K
    if value.startswith("2K") or "真实硬件" in value:
        return RES_2K_TO_2K
    return RES_1080_TO_1080


def load_config() -> dict[str, Any]:
    config = _new_parser()
    _ensure_schema(config)
    mode = config.get("Capture Methods", "source_mode", fallback="video").strip().lower()
    if mode not in LABEL_BY_MODE:
        mode = "video" if config.getboolean("Capture Methods", "simulation_mode", fallback=True) else "hardware"
    return {
        "source": LABEL_BY_MODE[mode],
        "source_path": config.get("Capture Methods", "source_path", fallback=DEFAULT_SOURCE_PATH),
        "debug_options": [
            label for label, option in DEBUG_OPTIONS.items()
            if config.getboolean("Debug window", option, fallback=True)
        ],
        "cooperative_filtering": config.getboolean("Control_Filter", "cooperative_filtering", fallback=True),
        "resolution_mode": normalize_res_mode(config.get("Control_Filter", "current_res_mode", fallback="1080P -> 2K")),
        "AI_conf": config.getfloat("AI", "AI_conf", fallback=0.2),
        "head_shot_ratio": config.getfloat("Aim", "head_shot_ratio", fallback=0.3),
        "mouse_sensitivity": config.getfloat("Mouse", "mouse_sensitivity", fallback=3.0),
    }


def _upsert_config_values(updates: dict[str, dict[str, str]]) -> None:
    """Update config.ini values while preserving comments and section ordering."""
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    current_section: str | None = None
    seen: dict[str, set[str]] = {section: set() for section in updates}
    inserted_sections: set[str] = set()

    def append_missing_for_section(section: str) -> None:
        if section not in updates or section in inserted_sections:
            return
        for key, value in updates[section].items():
            if key not in seen[section]:
                output.append(f"{key} = {value}")
                seen[section].add(key)
        inserted_sections.add(section)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_section is not None:
                append_missing_for_section(current_section)
            current_section = stripped[1:-1]
            output.append(line)
            continue

        if current_section in updates and "=" in line and not stripped.startswith(("#", ";")):
            key = line.split("=", 1)[0].strip()
            if key in updates[current_section]:
                output.append(f"{key} = {updates[current_section][key]}")
                seen[current_section].add(key)
                continue

        output.append(line)

    if current_section is not None:
        append_missing_for_section(current_section)

    for section, values in updates.items():
        if section not in inserted_sections:
            if output and output[-1].strip():
                output.append("")
            output.append(f"[{section}]")
            for key, value in values.items():
                output.append(f"{key} = {value}")

    CONFIG_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def write_config(
    source: str,
    source_path: str,
    debug_options: list[str],
    cooperative_filtering: bool,
    resolution_mode: str,
    ai_conf: float,
    head_shot_ratio: float,
    mouse_sensitivity: float,
) -> str:
    selected_debug = set(debug_options or [])
    debug_updates = {
        option: "True" if label in selected_debug else "False"
        for label, option in DEBUG_OPTIONS.items()
    }

    source_mode = MODE_BY_LABEL.get(source, "video")
    normalized_path = source_path.strip() or DEFAULT_SOURCE_PATH
    normalized_res_mode = normalize_res_mode(resolution_mode)
    res_matrix = RES_MATRIX_BY_MODE[normalized_res_mode]
    resolution_scale_factor = float(res_matrix["scale"])
    crop_width = int(res_matrix["crop_width"])
    crop_height = int(res_matrix["crop_height"])

    current_config = _new_parser()
    tag_threshold = current_config.get("Control_Filter", "tag_color_density_threshold", fallback="0.10")

    _upsert_config_values({
        "Detection window": {
            "detection_window_width": str(crop_width),
            "detection_window_height": str(crop_height),
        },
        "Capture Methods": {
            "source_mode": source_mode,
            "source_path": normalized_path,
            "simulation_mode": "True" if source_mode in {"video", "image"} else "False",
            "simulation_video_path": normalized_path,
        },
        "Debug window": debug_updates,
        "Control_Filter": {
            "cooperative_filtering": "True" if cooperative_filtering else "False",
            "tag_color_density_threshold": tag_threshold,
            "resolution_scale_factor": f"{resolution_scale_factor:.4f}".rstrip("0").rstrip("."),
            "current_res_mode": normalized_res_mode,
        },
        "AI": {"AI_conf": f"{float(ai_conf):.2f}"},
        "Aim": {"head_shot_ratio": f"{float(head_shot_ratio):.2f}"},
        "Mouse": {"mouse_sensitivity": f"{float(mouse_sensitivity):.2f}"},
    })

    return (
        "✅ 配置已写入 config.ini｜"
        f"source_mode={source_mode}, "
        f"source_path={normalized_path}, "
        f"debug={debug_updates}, "
        f"cooperative_filtering={bool(cooperative_filtering)}, "
        f"resolution_mode={normalized_res_mode}, "
        f"resolution_scale_factor={resolution_scale_factor:.4f}, "
        f"crop={crop_width}x{crop_height}, "
        f"AI_conf={float(ai_conf):.2f}, "
        f"head_shot_ratio={float(head_shot_ratio):.2f}, "
        f"mouse_sensitivity={float(mouse_sensitivity):.2f}"
    )


def process_status() -> str:
    with _process_lock:
        if _pipeline_process is None:
            return "未启动"
        code = _pipeline_process.poll()
        if code is None:
            return f"运行中｜PID={_pipeline_process.pid}"
        return f"已退出｜PID={_pipeline_process.pid}｜exit_code={code}"


def read_pipeline_log(max_lines: int = 160) -> str:
    """Return the tail of pipeline_debug.log for the live UI log panel."""
    log_path = PROJECT_ROOT / "pipeline_debug.log"
    if not log_path.exists():
        return "pipeline_debug.log 尚未生成。点击 Start Pipeline 后这里会实时显示日志。"

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"读取 pipeline_debug.log 失败：{exc}"

    lines = text.splitlines()
    tail = lines[-max_lines:]
    prefix = ""
    if len(lines) > max_lines:
        prefix = f"... 已省略前 {len(lines) - max_lines} 行 ...\n"
    return prefix + "\n".join(tail)


def start_pipeline(
    source: str,
    source_path: str,
    debug_options: list[str],
    cooperative_filtering: bool,
    resolution_mode: str,
    ai_conf: float,
    head_shot_ratio: float,
    mouse_sensitivity: float,
) -> str:
    global _pipeline_process, _pipeline_log_file
    config_msg = write_config(
        source,
        source_path,
        debug_options,
        cooperative_filtering,
        resolution_mode,
        ai_conf,
        head_shot_ratio,
        mouse_sensitivity,
    )

    with _process_lock:
        if _pipeline_process is not None and _pipeline_process.poll() is None:
            return f"{config_msg}\n⚠️ 核心管道已在运行：PID={_pipeline_process.pid}"

        if _pipeline_log_file is not None:
            _pipeline_log_file.close()
            _pipeline_log_file = None

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_path = PROJECT_ROOT / "pipeline_debug.log"
        _pipeline_log_file = log_path.open("w", encoding="utf-8")
        _pipeline_process = subprocess.Popen(
            [sys.executable, "run.py"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=_pipeline_log_file,
            stderr=_pipeline_log_file,
        )
        return (
            f"{config_msg}\n🚀 核心管道已启动：PID={_pipeline_process.pid}\n"
            f"日志文件：{log_path}"
        )


def terminate_pipeline() -> str:
    global _pipeline_process, _pipeline_log_file
    with _process_lock:
        if _pipeline_process is None:
            return "ℹ️ 当前没有由 Launcher 管理的核心管道进程。"

        code = _pipeline_process.poll()
        if code is not None:
            msg = f"ℹ️ 核心管道已退出：PID={_pipeline_process.pid}｜exit_code={code}"
            _pipeline_process = None
            if _pipeline_log_file is not None:
                _pipeline_log_file.close()
                _pipeline_log_file = None
            return msg

        pid = _pipeline_process.pid
        _pipeline_process.terminate()
        try:
            _pipeline_process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            # Fallback for stubborn GPU / camera handles.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGKILL)
            _pipeline_process.wait(timeout=5)

        _pipeline_process = None
        if _pipeline_log_file is not None:
            _pipeline_log_file.close()
            _pipeline_log_file = None
        return f"🛑 核心管道已终止：PID={pid}，GPU/摄像头资源已释放。"


def build_ui() -> gr.Blocks:
    defaults = load_config()

    with gr.Blocks(title="YOLOv8 Dashboard Launcher") as demo:
        gr.Markdown("# YOLOv8 Realtime Detection Dashboard", elem_id="title")
        gr.Markdown("轻量级参数固化、双源切换与核心推理进程生命周期管理", elem_id="subtitle")

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(
                    choices=SOURCE_LABELS,
                    value=defaults["source"],
                    label="Data Source Selector",
                )
                source_path = gr.Textbox(
                    label="仿真视频/图片绝对路径 (Source Path)",
                    value=defaults["source_path"],
                    placeholder=DEFAULT_SOURCE_PATH,
                )
                debug_options = gr.CheckboxGroup(
                    choices=list(DEBUG_OPTIONS.keys()),
                    value=defaults["debug_options"],
                    label="调试窗口渲染开关 (Debug Render Toggles)",
                )
                start_btn = gr.Button("🚀 初始化核心管道 (Start Pipeline)", variant="primary")
                stop_btn = gr.Button("🛑 终止系统进程 (Terminate Process)", variant="stop")
            with gr.Column(scale=2):
                cooperative_filtering = gr.Checkbox(
                    value=defaults["cooperative_filtering"],
                    label="开启协同目标身份清洗 (Cooperative Filtering)",
                )
                resolution_mode = gr.Dropdown(
                    choices=RES_MODE_CHOICES,
                    value=defaults["resolution_mode"],
                    label="全分辨率自适应映射 (Resolution Scale Matrix)",
                )
                ai_conf = gr.Slider(0.01, 1.0, value=defaults["AI_conf"], step=0.01, label="目标置信度阈值 (AI_conf)")
                head_ratio = gr.Slider(0.0, 1.0, value=defaults["head_shot_ratio"], step=0.05, label="多类别优先权权重 (head_shot_ratio)")
                sensitivity = gr.Slider(0.1, 10.0, value=defaults["mouse_sensitivity"], step=0.1, label="系统敏感度系数 (mouse_sensitivity)")

        status = gr.Textbox(label="Runtime Status", value=f"就绪｜进程状态：{process_status()}", lines=4, elem_classes=["status-box"])

        with gr.Accordion("📡 实时 Pipeline / Mouse Stream 日志", open=True):
            live_log = gr.Textbox(
                label="pipeline_debug.log live tail",
                value=read_pipeline_log(),
                lines=18,
                max_lines=18,
                interactive=False,
                elem_classes=["live-log"],
            )
            refresh_log_btn = gr.Button("🔄 手动刷新日志 (Refresh Log)")

        inputs = [source, source_path, debug_options, cooperative_filtering, resolution_mode, ai_conf, head_ratio, sensitivity]
        for component in inputs:
            component.change(write_config, inputs=inputs, outputs=status, show_progress=False)

        start_btn.click(start_pipeline, inputs=inputs, outputs=status, show_progress=True).then(
            read_pipeline_log,
            outputs=live_log,
            show_progress=False,
        )
        stop_btn.click(terminate_pipeline, outputs=status, show_progress=True).then(
            read_pipeline_log,
            outputs=live_log,
            show_progress=False,
        )
        refresh_log_btn.click(read_pipeline_log, outputs=live_log, show_progress=False)
        log_timer = gr.Timer(value=1.0, active=True)
        log_timer.tick(read_pipeline_log, outputs=live_log, show_progress=False)

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size="lg",
        ),
        css=DASHBOARD_CSS,
    )
