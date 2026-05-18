# YOLOv8 Realtime Detection Dashboard

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

## License

This project keeps the upstream MIT license. See `LICENSE` for details.
