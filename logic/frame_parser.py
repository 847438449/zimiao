import torch
import supervision as sv
import numpy as np
import random
import time
from typing import Sequence, Tuple

from logic.hotkeys_watcher import hotkeys_watcher
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse
from logic.shooting import shooting

class Target:
    def __init__(self, x, y, w, h, cls, focused=False):
        self.x = x
        self.y = y if focused or cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls

class DynamicFocusParser:
    """
    Dynamic focus selector for IVA/PTZ auto-tracking cameras.

    The parser chooses between two focus modes:
    - head: focus near the upper part of the bounding box
    - body: focus near the upper-body region of the bounding box

    A hysteresis lock is used to avoid high-frequency focus switching that may
    cause PTZ servo chattering.
    """

    HEAD_MODE = "head"
    BODY_MODE = "body"

    def __init__(self, lock_duration: float = 0.5) -> None:
        """
        Initialize the dynamic focus parser.

        Args:
            lock_duration: Minimum time in seconds before the focus mode may
                be resampled. Defaults to 0.5 seconds.

        Raises:
            ValueError: If lock_duration is negative.
        """
        if lock_duration < 0:
            raise ValueError("lock_duration must be non-negative")

        self.lock_duration = float(lock_duration)
        self.focus_mode = self.BODY_MODE
        self.last_switch_time = time.time()

    def update_focus_offset(
        self,
        box: Sequence[float],
        screen_center: Tuple[float, float] = (960, 540),
    ) -> Tuple[float, float]:
        """
        Calculate the selected focus offset relative to the screen center.

        Args:
            box: Bounding box in [x_min, y_min, x_max, y_max] format.
            screen_center: Reference center point of the screen, defaults to
                (960, 540).

        Returns:
            A tuple (dx, dy), representing the selected focus point offset from
            the screen center.

        Raises:
            ValueError: If box or screen_center is invalid.
        """
        x_min, y_min, x_max, y_max = self._validate_box(box)
        center_x, center_y = self._validate_screen_center(screen_center)

        now = time.time()

        if now - self.last_switch_time > self.lock_duration:
            self.focus_mode = (
                self.HEAD_MODE if random.random() < 0.3 else self.BODY_MODE
            )
            self.last_switch_time = now

        box_width = x_max - x_min
        box_height = y_max - y_min

        target_x = x_min + 0.5 * box_width

        if self.focus_mode == self.HEAD_MODE:
            target_y = y_min + 0.15 * box_height
        else:
            target_y = y_min + 0.42 * box_height

        dx = target_x - center_x
        dy = target_y - center_y

        return dx, dy

    @staticmethod
    def _validate_box(box: Sequence[float]) -> Tuple[float, float, float, float]:
        """
        Validate and normalize a bounding box.

        Args:
            box: Bounding box in [x_min, y_min, x_max, y_max] format.

        Returns:
            Normalized bounding box as floats.

        Raises:
            ValueError: If the box is malformed or has invalid geometry.
        """
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("box must be a list or tuple: [x_min, y_min, x_max, y_max]")

        try:
            x_min, y_min, x_max, y_max = map(float, box)
        except (TypeError, ValueError) as exc:
            raise ValueError("box coordinates must be numeric") from exc

        if x_max <= x_min:
            raise ValueError("box x_max must be greater than x_min")

        if y_max <= y_min:
            raise ValueError("box y_max must be greater than y_min")

        return x_min, y_min, x_max, y_max

    @staticmethod
    def _validate_screen_center(
        screen_center: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Validate and normalize the screen center.

        Args:
            screen_center: Screen center point as (x, y).

        Returns:
            Normalized screen center as floats.

        Raises:
            ValueError: If screen_center is malformed.
        """
        if not isinstance(screen_center, (list, tuple)) or len(screen_center) != 2:
            raise ValueError("screen_center must be a tuple or list: (x, y)")

        try:
            center_x, center_y = map(float, screen_center)
        except (TypeError, ValueError) as exc:
            raise ValueError("screen_center coordinates must be numeric") from exc

        return center_x, center_y

class FrameParser:
    def __init__(self):
        self.arch = self.get_arch()
        self.dynamic_focus_parser = DynamicFocusParser(lock_duration=0.5)

    def parse(self, result):
        if isinstance(result, sv.Detections):
            self._process_sv_detections(result)
        else:
            self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        if detections.xyxy.any():
            visuals.draw_helpers(detections)
            target = self.sort_targets(detections)
            self._handle_target(target)
        else:
            visuals.clear()
            if cfg.auto_shoot or cfg.triggerbot:
                shooting.shoot(False, False)

    def _process_yolo_detections(self, results):
        for frame in results:
            if frame.boxes:
                target = self.sort_targets(frame)
                self._handle_target(target)
                self._visualize_frame(frame)

    def _handle_target(self, target):
        if target:
            if hotkeys_watcher.clss is None:
                hotkeys_watcher.active_classes()

            if target.cls in hotkeys_watcher.clss:
                mouse.process_data((target.x, target.y, target.w, target.h, target.cls))

    def _visualize_frame(self, frame):
        if cfg.show_window or cfg.show_overlay:
            if cfg.show_boxes or cfg.overlay_show_boxes:
                visuals.draw_helpers(frame.boxes)

            if cfg.show_window and cfg.show_detection_speed:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])

        # Handle no detections
        if not frame.boxes and (cfg.auto_shoot or cfg.triggerbot):
            shooting.shoot(False, False)

        if cfg.show_window or cfg.show_overlay:
            if not frame.boxes:
                visuals.clear()

    def sort_targets(self, frame):
        if isinstance(frame, sv.Detections):
            boxes_array, classes_tensor = self._convert_sv_to_tensor(frame)
        else:
            boxes_array = frame.boxes.xywh.to(self.arch)
            classes_tensor = frame.boxes.cls.to(self.arch)

        if not classes_tensor.numel():
            return None

        return self._find_nearest_target(boxes_array, classes_tensor)

    def _convert_sv_to_tensor(self, frame):
        xyxy = frame.xyxy
        xywh = torch.tensor([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1]
        ], dtype=torch.float32).to(self.arch).T

        classes_tensor = torch.from_numpy(np.array(frame.class_id, dtype=np.float32)).to(self.arch)
        return xywh, classes_tensor

    def _find_nearest_target(self, boxes_array, classes_tensor):
        center = torch.tensor([capture.screen_x_center, capture.screen_y_center], device=self.arch)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        weights = torch.ones_like(distances_sq)

        if cfg.disable_headshot:
            non_head_mask = classes_tensor != 7
            weights = torch.ones_like(classes_tensor)
            weights[classes_tensor == 7] *= 0.5
            size_factor = boxes_array[:, 2] * boxes_array[:, 3]
            distances_sq = weights * (distances_sq / size_factor)

            if not non_head_mask.any():
                return None
            nearest_idx = torch.argmin(distances_sq[non_head_mask])
            nearest_idx = torch.nonzero(non_head_mask)[nearest_idx].item()
        else:
            head_mask = classes_tensor == 7
            if head_mask.any():
                nearest_idx = torch.argmin(distances_sq[head_mask])
                nearest_idx = torch.nonzero(head_mask)[nearest_idx].item()
            else:
                nearest_idx = torch.argmin(distances_sq)

        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = classes_tensor[nearest_idx].item()

        target_x, target_y, target_w, target_h = target_data
        x_min = target_x - target_w / 2
        y_min = target_y - target_h / 2
        x_max = target_x + target_w / 2
        y_max = target_y + target_h / 2
        screen_center = (capture.screen_x_center, capture.screen_y_center)

        dx, dy = self.dynamic_focus_parser.update_focus_offset(
            [x_min, y_min, x_max, y_max],
            screen_center=screen_center,
        )
        focused_x = screen_center[0] + dx
        focused_y = screen_center[1] + dy

        return Target(focused_x, focused_y, target_w, target_h, target_class, focused=True)

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        elif 'cpu' in cfg.AI_device:
            return 'cpu'
        else:
            return f'cuda:{cfg.AI_device}'

frameParser = FrameParser()
