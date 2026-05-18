# YOLOv8 Realtime Detection Dashboard

[中文说明](#中文说明) | [English](#english)

---

<a id="english"></a>

## English

A Windows-first realtime YOLO detection pipeline with a lightweight Gradio dashboard, dual-source capture switching, debug visualization toggles, and safer subprocess logging.

This fork is based on Sunone Aimbot, but the current working focus is a local YOLOv8/Ultralytics video-stream inference pipeline that can be tested from either:

- an offline simulation video file, or
- a UVC/USB hardware capture device.

> [!WARNING]
> This repository contains automation and aiming-related code inherited from the upstream project. Use responsibly and only in environments where you have permission. Online games may prohibit this type of software.

---

## Key Features

### Dashboard Launcher

`launcher_ui.py` provides a modern Gradio control panel for everyday operation.

It can:

- switch between offline video simulation and hardware capture mode;
- edit the simulation video path;
- tune inference/runtime parameters;
- toggle debug rendering options;
- start `run.py` as a managed subprocess;
- terminate the subprocess safely;
- write subprocess logs to `pipeline_debug.log`.

### Dual-source Capture Pipeline

`logic/capture.py` now reads `[Capture Methods]` from `config.ini`:

```ini
[Capture Methods]
simulation_mode = True
simulation_video_path = F:\yolo_training\game_test..mp4
```

Behavior:

- `simulation_mode = True`  
  Uses `cv2.VideoCapture(simulation_video_path)` for offline testing.
- `simulation_mode = False`  
  Uses the hardware capture stream, currently `cv2.VideoCapture(1)`.
- When the simulation video reaches EOF, the pipeline rewinds to frame `0` automatically and keeps inference running.

### Debug Window Controls

The dashboard can write these values into `[Debug window]` automatically:

```ini
show_window = True
show_boxes = True
show_labels = True
show_conf = True
```

No manual config editing is required for normal testing.

### Process Logging

When the dashboard starts the core pipeline, stdout and stderr are redirected to:

```text
pipeline_debug.log
```

This prevents silent crashes. If `run.py` exits early, check the last lines of this file.

---

## Requirements

Recommended environment:

- Windows 10/11
- Python 3.12
- NVIDIA GPU with CUDA-capable PyTorch for normal GPU inference
- Local virtual environment at `.venv`

Python packages are installed from:

```text
requirements.txt
```

Notable packages:

- `ultralytics`
- `torch`
- `opencv-python`
- `supervision`
- `gradio`

---

## Setup

From the project root:

```bat
setup_local.bat
```

This creates/updates `.venv`, installs dependencies, and runs a basic import check.

If you prefer manual setup:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt streamlit
.venv\Scripts\python.exe check_env.py
```

---

## Start the Dashboard

Recommended:

```bat
start_dashboard.bat
```

Or manually:

```bat
.venv\Scripts\python.exe launcher_ui.py
```

Then open:

```text
http://127.0.0.1:7860/
```

`start_dashboard.bat` also opens the browser automatically.

---

## Offline Video Test Workflow

1. Start the dashboard:

   ```bat
   start_dashboard.bat
   ```

2. In **Data Source Selector**, choose:

   ```text
   🎬 离线视频仿真 (Simulation Mode)
   ```

3. Set **仿真视频绝对路径 (Video Path)** to an existing local video, for example:

   ```text
   F:\yolo_training\game_test..mp4
   ```

4. Enable debug visualization if needed:

   - `show_window`
   - `show_boxes`
   - `show_labels`
   - `show_conf`

5. Tune parameters:

   - `AI_conf`
   - `head_shot_ratio`
   - `mouse_sensitivity`

6. Click:

   ```text
   🚀 初始化核心管道 (Start Pipeline)
   ```

7. If no window appears or the process exits, inspect:

   ```text
   pipeline_debug.log
   ```

8. To stop the pipeline, click:

   ```text
   🛑 终止系统进程 (Terminate Process)
   ```

---

## Hardware Capture Workflow

1. Connect the USB/UVC capture device.
2. Start the dashboard.
3. Choose:

   ```text
   📹 实时硬件采集 (Hardware Stream Mode)
   ```

4. Click **Start Pipeline**.

The default hardware capture index is configured in `logic/capture.py`:

```python
USB_CAPTURE_INDEX = 1
USB_CAPTURE_WIDTH = 1920
USB_CAPTURE_HEIGHT = 1080
USB_CAPTURE_FPS = 60
```

Adjust these constants if your device uses a different index or resolution.

---

## Important Files

```text
launcher_ui.py          Gradio dashboard and subprocess manager
start_dashboard.bat     One-click dashboard launcher
run.py                  Core inference loop
logic/capture.py        Dual-source frame capture pipeline
logic/config_watcher.py config.ini reader
logic/frame_parser.py   Detection target selection logic
logic/checks.py         Environment/model validation
config.ini              Runtime configuration
pipeline_debug.log      Generated runtime log; safe to delete
```

---

## Configuration Notes

### Model Path

`config.ini` uses:

```ini
[AI]
AI_model_name = best.pt
```

The code checks both:

```text
models/best.pt
best.pt
```

So the model can be placed either in `models/` or in the project root.

### Simulation Video Path

The simulation video path is controlled by:

```ini
[Capture Methods]
simulation_video_path = F:\yolo_training\game_test..mp4
```

The dashboard writes this field automatically when the path textbox changes.

### Debug Window

If the program is running but no visual output appears, enable these from the dashboard:

```ini
[Debug window]
show_window = True
show_boxes = True
show_labels = True
show_conf = True
```

---

## Troubleshooting

### Dashboard starts but pipeline immediately stops

Check:

```text
pipeline_debug.log
```

Common causes:

- missing model file;
- wrong video path;
- CUDA/PyTorch mismatch;
- missing dependency;
- unavailable camera index.

### `python` uses the wrong environment

The dashboard uses `sys.executable` when starting `run.py`, so it should use the same `.venv` interpreter that launched `launcher_ui.py`.

Always start the dashboard with:

```bat
.venv\Scripts\python.exe launcher_ui.py
```

or:

```bat
start_dashboard.bat
```

### Video does not open

Confirm the file exists and the path in the dashboard is exact. Windows paths should look like:

```text
F:\folder\video.mp4
```

---

## Legacy Launchers

The original helper launchers are still available:

```bat
run_ai.bat       Start run.py directly
run_helper.bat   Start the Streamlit helper UI
```

For the new workflow, prefer:

```bat
start_dashboard.bat
```

---

<a id="中文说明"></a>

## 中文说明

这是一个面向 Windows 的 YOLO 实时视频流目标检测项目，加入了轻量级 Gradio Dashboard、双数据源切换、调试窗口可视化开关，以及后台子进程日志记录能力。

本项目基于 Sunone Aimbot 修改，但当前重点是构建一个可用于本地测试和硬件采集的 YOLOv8 / Ultralytics 实时推理流水线。

支持两种输入源：

- 本地离线视频仿真测试；
- UVC / USB 硬件视频采集设备。

> [!WARNING]
> 本仓库继承了上游项目中的自动化和瞄准相关代码。请只在你有权限的环境中负责任地使用。很多在线游戏禁止此类软件。

---

## 核心功能

### Gradio 控制面板

`launcher_ui.py` 提供一个现代化网页控制台，用于日常启动和调参。

它可以：

- 在离线视频仿真和硬件采集之间切换；
- 修改仿真视频绝对路径；
- 调整推理和运行参数；
- 开关调试窗口、检测框、类别标签、置信度显示；
- 以受控子进程方式启动 `run.py`；
- 安全终止核心推理进程；
- 将后台运行日志写入 `pipeline_debug.log`。

### 双源采集底层流水线

`logic/capture.py` 会读取 `config.ini` 中的 `[Capture Methods]`：

```ini
[Capture Methods]
simulation_mode = True
simulation_video_path = F:\yolo_training\game_test..mp4
```

运行逻辑：

- `simulation_mode = True`  
  使用 `cv2.VideoCapture(simulation_video_path)` 读取本地视频进行仿真测试。
- `simulation_mode = False`  
  使用硬件采集流，当前默认是 `cv2.VideoCapture(1)`。
- 当仿真视频播放到结尾时，程序会自动跳回第 `0` 帧，实现循环回放，保证推理流水线持续运行。

### 调试窗口开关

Dashboard 可以自动写入 `[Debug window]`：

```ini
show_window = True
show_boxes = True
show_labels = True
show_conf = True
```

正常测试时不需要手动修改 `config.ini`。

### 后台日志记录

通过 Dashboard 启动核心推理程序时，`stdout` 和 `stderr` 会被重定向到：

```text
pipeline_debug.log
```

这样可以避免“分配了 PID 但程序没有响应”的沉默崩溃问题。如果 `run.py` 闪退，直接看这个文件最后几行即可定位错误。

---

## 环境要求

推荐环境：

- Windows 10 / 11
- Python 3.12
- NVIDIA GPU + 支持 CUDA 的 PyTorch，用于正常 GPU 推理
- 项目根目录下存在 `.venv` 虚拟环境

依赖安装来自：

```text
requirements.txt
```

主要依赖包括：

- `ultralytics`
- `torch`
- `opencv-python`
- `supervision`
- `gradio`

---

## 安装 / 初始化

在项目根目录运行：

```bat
setup_local.bat
```

这个脚本会创建或更新 `.venv`，安装依赖，并执行基础导入检查。

如果想手动安装：

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt streamlit
.venv\Scripts\python.exe check_env.py
```

---

## 启动 Dashboard

推荐直接运行：

```bat
start_dashboard.bat
```

或者手动运行：

```bat
.venv\Scripts\python.exe launcher_ui.py
```

然后打开浏览器访问：

```text
http://127.0.0.1:7860/
```

`start_dashboard.bat` 会自动尝试打开浏览器页面。

---

## 离线视频测试流程

1. 启动 Dashboard：

   ```bat
   start_dashboard.bat
   ```

2. 在 **Data Source Selector** 中选择：

   ```text
   🎬 离线视频仿真 (Simulation Mode)
   ```

3. 在 **仿真视频绝对路径 (Video Path)** 中填写实际存在的视频路径，例如：

   ```text
   F:\yolo_training\game_test..mp4
   ```

4. 根据需要勾选调试显示：

   - `show_window`：显示渲染窗口；
   - `show_boxes`：显示目标边界框；
   - `show_labels`：显示类别标签；
   - `show_conf`：显示置信度数值。

5. 调整参数：

   - `AI_conf`：目标置信度阈值；
   - `head_shot_ratio`：Class 0 / Class 1 多类别优先权权重；
   - `mouse_sensitivity`：系统敏感度系数。

6. 点击：

   ```text
   🚀 初始化核心管道 (Start Pipeline)
   ```

7. 如果没有弹窗或程序退出，查看：

   ```text
   pipeline_debug.log
   ```

8. 停止测试时点击：

   ```text
   🛑 终止系统进程 (Terminate Process)
   ```

---

## 硬件采集流程

1. 连接 USB / UVC 采集设备。
2. 启动 Dashboard。
3. 选择：

   ```text
   📹 实时硬件采集 (Hardware Stream Mode)
   ```

4. 点击 **Start Pipeline**。

默认硬件采集参数位于 `logic/capture.py`：

```python
USB_CAPTURE_INDEX = 1
USB_CAPTURE_WIDTH = 1920
USB_CAPTURE_HEIGHT = 1080
USB_CAPTURE_FPS = 60
```

如果你的设备索引或分辨率不同，可以修改这些常量。

---

## 重要文件说明

```text
launcher_ui.py          Gradio Dashboard 和子进程管理器
start_dashboard.bat     一键启动 Dashboard 的 Windows 脚本
run.py                  核心推理循环
logic/capture.py        双源视频帧采集模块
logic/config_watcher.py config.ini 配置读取模块
logic/frame_parser.py   检测结果目标选择逻辑
logic/checks.py         环境与模型检查
config.ini              运行时配置文件
pipeline_debug.log      运行时生成的调试日志，可删除
```

---

## 配置说明

### 模型路径

`config.ini` 中使用：

```ini
[AI]
AI_model_name = best.pt
```

代码会依次检查：

```text
models/best.pt
best.pt
```

所以模型可以放在 `models/` 目录，也可以放在项目根目录。

### 仿真视频路径

仿真视频路径由以下配置控制：

```ini
[Capture Methods]
simulation_video_path = F:\yolo_training\game_test..mp4
```

当你在 Dashboard 的路径输入框中修改内容时，这个字段会自动写回 `config.ini`。

### 调试窗口

如果程序正在运行但没有可视窗口，可以在 Dashboard 中打开：

```ini
[Debug window]
show_window = True
show_boxes = True
show_labels = True
show_conf = True
```

---

## 常见问题排查

### Dashboard 能启动，但核心管道马上退出

查看：

```text
pipeline_debug.log
```

常见原因：

- 模型文件缺失；
- 视频路径错误；
- CUDA / PyTorch 版本不匹配；
- 依赖缺失；
- 摄像头 / 采集卡索引不可用。

### Python 环境用错

Dashboard 使用 `sys.executable` 启动 `run.py`，因此会使用启动 `launcher_ui.py` 的同一个 `.venv` 解释器。

推荐始终使用：

```bat
start_dashboard.bat
```

或：

```bat
.venv\Scripts\python.exe launcher_ui.py
```

### 视频打不开

请确认视频文件真实存在，且 Dashboard 中填写的是完整 Windows 路径，例如：

```text
F:\folder\video.mp4
```

---

## 旧启动脚本

原来的启动方式仍保留：

```bat
run_ai.bat       直接启动 run.py
run_helper.bat   启动 Streamlit 配置助手
```

新的推荐流程是：

```bat
start_dashboard.bat
```

---

## License

This project keeps the upstream MIT license. See `LICENSE` for details.
