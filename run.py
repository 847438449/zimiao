from __future__ import annotations

from pathlib import Path
import queue

from ultralytics import YOLO
import cv2
import torch

from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.frame_parser import frameParser
from logic.hotkeys_watcher import hotkeys_watcher
from logic.checks import run_checks
import supervision as sv
    
# Debug/control-stability mode: force pure single-frame servo, no ByteTrack state lag.
cfg.disable_tracker = True
tracker = None
last_frame_signature: bytes | None = None


def build_frame_signature(inference_frame) -> bytes:
    """Return a low-cost ROI fingerprint for frame-lock deduplication."""
    if inference_frame is None or inference_frame.size == 0:
        return b""
    tiny = cv2.resize(inference_frame, (16, 16), interpolation=cv2.INTER_AREA)
    return tiny.tobytes()


def enqueue_visual_frame(image) -> None:
    """Keep visualization best-effort and never block the control loop."""
    try:
        if visuals.queue.full():
            try:
                visuals.queue.get_nowait()
            except queue.Empty:
                pass
        visuals.queue.put_nowait(image)
    except queue.Full:
        pass


@torch.inference_mode()
def perform_detection(model, image, tracker: sv.ByteTrack | None = None):
    kwargs = dict(
        source=image,
        imgsz=cfg.ai_model_image_size,
        conf=cfg.AI_conf,
        iou=0.50,
        device=cfg.AI_device,
        half=not "cpu" in cfg.AI_device,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False,
        stream=True
    )

    kwargs["cfg"] = "logic/tracker.yaml" if tracker else "logic/game.yaml"

    results = model.predict(**kwargs)

    if tracker:
        for res in results:
            det = sv.Detections.from_ultralytics(res)
            return tracker.update_with_detections(det)
    else:
        return next(results)

def init():
    global last_frame_signature
    run_checks()
    
    try:
        model_path = Path("models") / cfg.AI_model_name
        if not model_path.exists():
            model_path = Path(cfg.AI_model_name)
        model = YOLO(str(model_path), task="detect")
    except Exception as e:
        print("An error occurred when loading the AI model:\n", e)
        quit(0)
        
    while True:
        full_frame = capture.get_new_frame()

        if full_frame is not None:
            image = frameParser.prepare_inference_frame(full_frame)

            if cfg.circle_capture:
                image = capture.convert_to_circle(image)

            current_signature = build_frame_signature(image)
            if last_frame_signature == current_signature:
                # Frame-Lock Synchronization Gate: skip stale OBS/cache duplicates
                # using the active ROI, not only a tiny full-frame center patch.
                continue
            last_frame_signature = current_signature

            if cfg.show_window or cfg.show_overlay:
                enqueue_visual_frame(image)

            result = perform_detection(model, image, tracker)

            if hotkeys_watcher.app_pause == 0:
                frameParser.parse(result, current_frame=image)

if __name__ == "__main__":
    init()