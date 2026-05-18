import torch
import supervision as sv
import numpy as np
import random
import time
import cv2
import configparser
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
        self.config = configparser.ConfigParser()
        self.reload_runtime_config()

    def reload_runtime_config(self):
        """Hot-read config.ini so UI profile changes affect the next frame."""
        self.config.clear()
        self.config.read("config.ini", encoding="utf-8")
        return self.config

    def parse(self, result, current_frame=None):
        if isinstance(result, sv.Detections):
            self._process_sv_detections(result, current_frame=current_frame)
        else:
            self._process_yolo_detections(result, current_frame=current_frame)

    def _process_sv_detections(self, detections, current_frame=None):
        if detections.xyxy.any():
            visuals.draw_helpers(detections)
            target = self.sort_targets(detections, current_frame=current_frame)
            self._handle_target(target)
        else:
            visuals.clear()
            if cfg.auto_shoot or cfg.triggerbot:
                shooting.shoot(False, False)

    def _process_yolo_detections(self, results, current_frame=None):
        for frame in results:
            if frame.boxes:
                target = self.sort_targets(frame, current_frame=current_frame)
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

    def sort_targets(self, frame, current_frame=None):
        if isinstance(frame, sv.Detections):
            boxes_array, classes_tensor = self._convert_sv_to_tensor(frame, current_frame=current_frame)
        else:
            boxes_array, classes_tensor = self._convert_yolo_to_tensor(frame, current_frame=current_frame)

        if not classes_tensor.numel():
            return None

        return self.route_targets_by_environment_profile(boxes_array, classes_tensor)

    def route_targets_by_environment_profile(self, boxes_array, classes_tensor):
        """Apply the multi-stage profile router and return the final target."""
        self.reload_runtime_config()
        active_profile = self.config.get(
            "Environment_Profile",
            "current_profile",
            fallback="profile_a",
        ).strip().lower()

        if active_profile == "profile_b":
            selected_class = self.config.get(
                "Control_Filter",
                "active_target_category",
                fallback="Category_A",
            ).strip().upper()
            strategy = self.config.get(
                "Control_Filter",
                "feature_convergence_strategy",
                fallback="strategy_first",
            ).strip().lower()

            if selected_class == "CATEGORY_B":
                primary_cls, secondary_cls = 3, 1
            else:
                primary_cls, secondary_cls = 2, 0

            detection_rows = [
                (boxes_array[idx], classes_tensor[idx])
                for idx in range(int(classes_tensor.numel()))
            ]
            active_keypoints = [
                row for row in detection_rows
                if int(row[1].item()) == primary_cls
            ]
            active_centroids = [
                row for row in detection_rows
                if int(row[1].item()) == secondary_cls
            ]

            if strategy == "keypoint_only":
                return self.get_closest_target(active_keypoints)
            if strategy == "centroid_only":
                return self.get_closest_target(active_centroids)

            final_servo_target = self.get_closest_target(active_keypoints)
            if final_servo_target is None:
                final_servo_target = self.get_closest_target(active_centroids)
            return final_servo_target

        standard_pool = [
            (boxes_array[idx], classes_tensor[idx])
            for idx in range(int(classes_tensor.numel()))
            if int(classes_tensor[idx].item()) == 0
        ]
        return self.get_closest_target(standard_pool)

    def get_closest_target(self, detection_pool):
        """Select the nearest candidate from a pre-filtered detection pool."""
        if not detection_pool:
            return None

        boxes = torch.stack([row[0] for row in detection_pool]).to(self.arch)
        classes = torch.stack([row[1] for row in detection_pool]).to(self.arch)
        return self._find_nearest_target(boxes, classes)

    def _convert_sv_to_tensor(self, frame, current_frame=None):
        xyxy = np.asarray(frame.xyxy)
        class_ids = np.array(frame.class_id if frame.class_id is not None else [], dtype=np.float32)
        xyxy, class_ids = self._filter_cooperative_candidates(current_frame, xyxy, class_ids)

        if xyxy.size == 0 or class_ids.size == 0:
            return (
                torch.empty((0, 4), dtype=torch.float32, device=self.arch),
                torch.empty((0,), dtype=torch.float32, device=self.arch),
            )

        xywh = torch.tensor([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1]
        ], dtype=torch.float32).to(self.arch).T

        classes_tensor = torch.from_numpy(class_ids.astype(np.float32)).to(self.arch)
        return xywh, classes_tensor

    def _convert_yolo_to_tensor(self, frame, current_frame=None):
        xyxy = frame.boxes.xyxy.detach().cpu().numpy()
        class_ids = frame.boxes.cls.detach().cpu().numpy().astype(np.float32)
        xyxy, class_ids = self._filter_cooperative_candidates(current_frame, xyxy, class_ids)

        if xyxy.size == 0 or class_ids.size == 0:
            return (
                torch.empty((0, 4), dtype=torch.float32, device=self.arch),
                torch.empty((0,), dtype=torch.float32, device=self.arch),
            )

        xywh = torch.tensor([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1]
        ], dtype=torch.float32).to(self.arch).T
        classes_tensor = torch.from_numpy(class_ids.astype(np.float32)).to(self.arch)
        return xywh, classes_tensor

    def read_filter_runtime(self):
        try:
            import configparser
            parser = configparser.ConfigParser()
            parser.read("config.ini", encoding="utf-8")
            anti_team_kill = parser.getboolean("Aim", "anti_team_kill", fallback=getattr(cfg, "anti_team_kill", True))
            cooperative = parser.getboolean("Control_Filter", "cooperative_filtering", fallback=getattr(cfg, "cooperative_filtering", True))
            threshold = parser.getfloat(
                "Aim",
                "teammate_color_threshold",
                fallback=parser.getfloat("Control_Filter", "tag_color_density_threshold", fallback=0.10),
            )
            return anti_team_kill and cooperative, threshold
        except Exception:
            return getattr(cfg, "anti_team_kill", True) and getattr(cfg, "cooperative_filtering", False), getattr(cfg, "tag_color_density_threshold", 0.10)

    def _filter_cooperative_candidates(self, current_frame, xyxy, class_ids):
        filter_enabled, _ = self.read_filter_runtime()
        if not filter_enabled or current_frame is None or xyxy.size == 0:
            return xyxy, class_ids

        keep_indices = []
        for idx, box_xyxy in enumerate(xyxy):
            if self.validate_cooperative_entity(current_frame, box_xyxy):
                print(
                    "[Mouse Stream] Target: 🛡️ COOP | "
                    "识别到协同保护目标，当前控制回路执行非合作化清洗 | Mode: SKIP",
                    flush=True,
                )
                continue
            keep_indices.append(idx)

        if not keep_indices:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return xyxy[keep_indices], class_ids[keep_indices]

    def validate_cooperative_entity(self, frame, bounding_box, roi_offset_height=30):
        """
        Detect active cooperative color tags in an ROI above the target box.

        Returns True when high-saturation green/blue UI tag pixels exceed the
        configured density threshold, causing the control loop to skip target.
        """
        try:
            if frame is None:
                return False

            if hasattr(bounding_box, "xyxy"):
                xyxy = bounding_box.xyxy[0].detach().cpu().numpy()
            else:
                xyxy = np.asarray(bounding_box, dtype=np.float32)

            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            height, width = frame.shape[:2]
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))

            roi_y1 = max(0, y1 - roi_offset_height)
            roi_y2 = y1

            if roi_y2 <= roi_y1 or x2 <= x1:
                return False

            feature_roi = frame[roi_y1:roi_y2, x1:x2]
            total_pixels = feature_roi.shape[0] * feature_roi.shape[1]
            if total_pixels == 0:
                return False

            hsv_roi = cv2.cvtColor(feature_roi, cv2.COLOR_BGR2HSV)

            lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
            lower_blue, upper_blue = np.array([100, 40, 40]), np.array([140, 255, 255])

            mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
            mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            combined_mask = cv2.bitwise_or(mask_green, mask_blue)

            activated_pixels = cv2.countNonZero(combined_mask)
            color_density_ratio = activated_pixels / total_pixels
            _, cfg_threshold = self.read_filter_runtime()
            return color_density_ratio > float(cfg_threshold)
        except Exception:
            return False

    def _find_nearest_target(self, boxes_array, classes_tensor):
        center = torch.tensor([capture.screen_x_center, capture.screen_y_center], device=self.arch)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        weights = torch.ones_like(distances_sq)

        class01_mask = (classes_tensor == 0) | (classes_tensor == 1)
        if class01_mask.any():
            # head_shot_ratio is exposed in launcher_ui.py as a runtime tuning knob.
            # Higher values favor Class 0; lower values favor Class 1 while distance
            # to the capture center remains the primary ranking signal.
            ratio = max(0.0, min(1.0, float(getattr(cfg, "head_shot_ratio", 0.3))))
            class_weights = torch.ones_like(distances_sq)
            class_weights[classes_tensor == 0] *= 1.0 - 0.5 * ratio
            class_weights[classes_tensor == 1] *= 1.0 - 0.5 * (1.0 - ratio)
            weighted_distances = distances_sq * class_weights
            nearest_idx = torch.argmin(weighted_distances[class01_mask])
            nearest_idx = torch.nonzero(class01_mask)[nearest_idx].item()
        elif cfg.disable_headshot:
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
