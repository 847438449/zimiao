from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO
import torch

from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.frame_parser import frameParser
from logic.hotkeys_watcher import hotkeys_watcher
from logic.checks import run_checks
import supervision as sv
    
tracker = sv.ByteTrack() if not cfg.disable_tracker else None

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
        image = capture.get_new_frame()
        
        if image is not None:
            if cfg.circle_capture:
                image = capture.convert_to_circle(image)
                
            if cfg.show_window or cfg.show_overlay:
                visuals.queue.put(image)
                
            result = perform_detection(model, image, tracker)

            if hotkeys_watcher.app_pause == 0:
                frameParser.parse(result, current_frame=image)

if __name__ == "__main__":
    init()