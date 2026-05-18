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
    def __init__(self, x, y, w, h, cls, focused=False, velocity_abs=0.0):
        self.x = x
        self.y = y if focused or cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls
        self.velocity_abs = float(velocity_abs)

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
        self.config_last_read = 0.0
        self.config_reload_interval = 0.10
        self.last_tracked_center = None
        self.last_tracked_target = None
        self.lost_frame_count = 0
        self.max_lost_frames = 5
        self.lock_radius = 45.0
        self.is_monopoly_locked = False
        self.locked_target_id = None
        self.last_locked_box = None
        self.lock_lost_frames = 0
        self.max_lost_window = 5
        self.locking_hysteresis_radius = 60.0
        self.prev_target_center = None
        self.smoothed_target_center = None
        self.anchor_smoothing_alpha = 0.35
        self.feedforward_deadzone_pixels = 2.0
        self.feedforward_y_deadzone_pixels = 4.0
        self.dt = 1.0
        self.prediction_factor = 1.5
        self.prediction_factor_y = 0.25
        self.max_feedforward_velocity = 20.0
        self.max_feedforward_velocity_y = 6.0
        self.prev_error_norm = None
        self.last_anchor_class = None
        self.reload_runtime_config(force=True)

    def reload_runtime_config(self, force=False):
        """Hot-read config.ini with a short TTL to avoid per-frame disk jitter."""
        now = time.monotonic()
        if (not force) and (now - self.config_last_read < self.config_reload_interval):
            return self.config
        self.config.clear()
        self.config.read("config.ini", encoding="utf-8")
        self.config_last_read = now
        return self.config

    def prepare_inference_frame(self, full_frame):
        """Crop the configured center ROI from a full-resolution frame for inference."""
        self.reload_runtime_config()
        target_w = self.config.getint("Detection window", "detection_window_width", fallback=448)
        target_h = self.config.getint("Detection window", "detection_window_height", fallback=448)
        target_w = max(32, int(target_w))
        target_h = max(32, int(target_h))

        capture.update_detection_window(target_w, target_h)

        img_h, img_w = full_frame.shape[:2]
        cx, cy = img_w // 2, img_h // 2
        x1 = cx - (target_w // 2)
        y1 = cy - (target_h // 2)
        x2 = x1 + target_w
        y2 = y1 + target_h

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)
        roi = full_frame[src_y1:src_y2, src_x1:src_x2]

        if roi.shape[1] == target_w and roi.shape[0] == target_h:
            return roi.copy()

        padded = np.zeros((target_h, target_w, full_frame.shape[2]), dtype=full_frame.dtype)
        dst_x1 = max(0, -x1)
        dst_y1 = max(0, -y1)
        dst_x2 = dst_x1 + roi.shape[1]
        dst_y2 = dst_y1 + roi.shape[0]
        padded[dst_y1:dst_y2, dst_x1:dst_x2] = roi
        return padded

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
                mouse.process_data((target.x, target.y, target.w, target.h, target.cls, target.velocity_abs))

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
            return self._handle_tracking_miss()

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
            monopoly_handled, monopoly_target = self.try_monopoly_short_circuit(detection_rows)
            if monopoly_handled:
                return monopoly_target
            active_keypoints = [
                row for row in detection_rows
                if self._row_class(row) == primary_cls
            ]
            active_centroids = [
                row for row in detection_rows
                if self._row_class(row) == secondary_cls
            ]

            if strategy == "keypoint_only":
                return self.select_target_with_hysteresis(active_keypoints)
            if strategy == "centroid_only":
                return self.select_target_with_hysteresis(active_centroids)

            if active_keypoints:
                primary_target = self.select_target_with_hysteresis(active_keypoints)
                if primary_target is not None:
                    return primary_target
            return self.select_target_with_hysteresis(active_centroids)

        standard_pool = [
            (boxes_array[idx], classes_tensor[idx])
            for idx in range(int(classes_tensor.numel()))
            if int(round(float(classes_tensor[idx].item()))) in {0, 1, 7}
        ]
        monopoly_handled, monopoly_target = self.try_monopoly_short_circuit(standard_pool)
        if monopoly_handled:
            return monopoly_target
        routed_pool = self.apply_hierarchical_label_routing(standard_pool)
        if not routed_pool:
            routed_pool = standard_pool
        return self.select_target_with_hysteresis(routed_pool)

    def _row_class(self, row):
        return int(round(float(row[1].detach().float().cpu().item())))

    def _profile_head_classes(self):
        active_profile = self.config.get("Environment_Profile", "current_profile", fallback="profile_a").strip().lower()
        if active_profile == "profile_b":
            selected_class = self.config.get("Control_Filter", "active_target_category", fallback="Category_A").strip().upper()
            return {3} if selected_class == "CATEGORY_B" else {2}
        return {7}

    def _profile_body_classes(self):
        active_profile = self.config.get("Environment_Profile", "current_profile", fallback="profile_a").strip().lower()
        if active_profile == "profile_b":
            selected_class = self.config.get("Control_Filter", "active_target_category", fallback="Category_A").strip().upper()
            return {1} if selected_class == "CATEGORY_B" else {0}
        return {0, 1}

    def apply_hierarchical_label_routing(self, detection_rows):
        """Strict but safe Head-over-Body routing; never return an empty candidate set."""
        if not detection_rows:
            return detection_rows

        head_classes = self._profile_head_classes()
        body_classes = self._profile_body_classes()
        heads = [row for row in detection_rows if self._row_class(row) in head_classes]
        bodies = [row for row in detection_rows if self._row_class(row) in body_classes]
        if not heads:
            return detection_rows
        if not bodies:
            return heads or detection_rows

        # Head labels are terminal precision anchors.  Return the Head pool itself
        # instead of “all non-body rows” so Body can never re-enter downstream NN.
        return heads or detection_rows

    def _body_has_head_override(self, body_row, head_rows, margin=50.0):
        body_box = body_row[0].detach().float().cpu().numpy()
        bx, by, bw, bh = [float(v) for v in body_box[:4]]
        bx1, by1 = bx - bw / 2.0 - margin, by - bh / 2.0 - margin
        bx2, by2 = bx + bw / 2.0 + margin, by + bh / 2.0 + margin

        for head_row in head_rows:
            head_box = head_row[0].detach().float().cpu().numpy()
            hx, hy, hw, hh = [float(v) for v in head_box[:4]]
            hx1, hy1 = hx - hw / 2.0, hy - hh / 2.0
            hx2, hy2 = hx + hw / 2.0, hy + hh / 2.0

            center_inside = bx1 <= hx <= bx2 and by1 <= hy <= by2
            inter_w = max(0.0, min(bx2, hx2) - max(bx1, hx1))
            inter_h = max(0.0, min(by2, hy2) - max(by1, hy1))
            inter_area = inter_w * inter_h
            head_area = max(1e-6, hw * hh)
            head_overlap_ratio = inter_area / head_area

            if center_inside or head_overlap_ratio > 0.0:
                return True
        return False

    def try_monopoly_short_circuit(self, detection_pool):
        """Highest-priority monopoly state; handled=True blocks blind Head/Body reroute."""
        if not self.is_monopoly_locked or self.last_locked_box is None:
            return False, None
        if self.lock_lost_frames >= self.max_lost_window:
            self._reset_tracking_state()
            return False, None
        if not detection_pool:
            return True, self._handle_tracking_miss()

        locked_cls = int(self.locked_target_id) if self.locked_target_id is not None else None
        if locked_cls in self._profile_body_classes():
            heads = [row for row in detection_pool if self._row_class(row) in self._profile_head_classes()]
            if self._locked_body_has_head_override(heads):
                # Break Body monopoly immediately so the hierarchy router can promote Head/keypoint.
                self.is_monopoly_locked = False
                self.last_locked_box = None
                self.locked_target_id = None
                self.lock_lost_frames = 0
                return False, None

        locked_row = self._find_spatially_continuous_row(detection_pool)
        if locked_row is None:
            return True, self._handle_tracking_miss()

        target = self.get_closest_target([locked_row])
        self._update_tracking_state(target, source_row=locked_row)
        return True, target

    def select_target_with_hysteresis(self, detection_pool):
        """Select a target with sticky spatial continuity before global search."""
        if not detection_pool:
            return self._handle_tracking_miss()

        if self._should_force_head_takeover(detection_pool):
            target_row = self._find_globally_nearest_row(detection_pool)
            target = self.get_closest_target([target_row]) if target_row is not None else None
            self._update_tracking_state(target, source_row=target_row)
            return target

        if self.last_locked_box is not None and self.lock_lost_frames < self.max_lost_window:
            locked_row = self._find_spatially_continuous_row(detection_pool)
            if locked_row is not None:
                target = self.get_closest_target([locked_row])
                self._update_tracking_state(target, source_row=locked_row)
                return target

            # Short-term miss state: stay silent and do not fall back to blind global reselection.
            return self._handle_tracking_miss()

        target_row = self._find_globally_nearest_row(detection_pool)
        target = self.get_closest_target([target_row]) if target_row is not None else None
        self._update_tracking_state(target, source_row=target_row)
        return target

    def _box_center_from_xywh(self, box_tensor):
        box = box_tensor.detach().float().cpu().numpy()
        return float(box[0]), float(box[1])

    def _should_force_head_takeover(self, detection_pool):
        """Allow a local Head/keypoint anchor to immediately break an existing Body lock."""
        if self.locked_target_id is None or self.last_locked_box is None:
            return False
        if int(self.locked_target_id) not in self._profile_body_classes():
            return False
        heads = [row for row in detection_pool if self._row_class(row) in self._profile_head_classes()]
        return self._locked_body_has_head_override(heads)

    def _locked_body_has_head_override(self, head_rows, margin=50.0):
        if not head_rows or self.last_locked_box is None:
            return False
        bx, by, bw, bh = [float(v) for v in self.last_locked_box[:4]]
        bx1, by1 = bx - bw / 2.0 - margin, by - bh / 2.0 - margin
        bx2, by2 = bx + bw / 2.0 + margin, by + bh / 2.0 + margin
        for head_row in head_rows:
            hx, hy = [float(v) for v in head_row[0][:2].detach().float().cpu().numpy()]
            if bx1 <= hx <= bx2 and by1 <= hy <= by2:
                return True
        return False

    def _is_class_compatible_with_lock(self, row):
        if self.locked_target_id is None:
            return True
        locked_cls = int(self.locked_target_id)
        row_cls = self._row_class(row)
        if locked_cls in self._profile_head_classes():
            return row_cls in self._profile_head_classes()
        return row_cls == locked_cls

    def _find_spatially_continuous_row(self, detection_pool):
        if self.last_locked_box is None:
            return None

        last_center = torch.tensor(
            [float(self.last_locked_box[0]), float(self.last_locked_box[1])],
            dtype=torch.float32,
            device=self.arch,
        )
        candidates = []
        for row in detection_pool:
            if not self._is_class_compatible_with_lock(row):
                continue
            center = row[0][:2].to(self.arch)
            distance = torch.linalg.vector_norm(center - last_center).item()
            if distance < self.locking_hysteresis_radius:
                candidates.append((distance, row))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _find_globally_nearest_row(self, detection_pool):
        if not detection_pool:
            return None
        roi_center_x = float(self.config.getint("Detection window", "detection_window_width", fallback=448)) / 2.0
        roi_center_y = float(self.config.getint("Detection window", "detection_window_height", fallback=448)) / 2.0
        center = torch.tensor([roi_center_x, roi_center_y], dtype=torch.float32, device=self.arch)
        return min(
            detection_pool,
            key=lambda row: torch.linalg.vector_norm(row[0][:2].to(self.arch) - center).item(),
        )

    def _handle_tracking_miss(self):
        """Hold the sticky lock silently, then reset after consecutive misses."""
        if self.last_locked_box is None:
            return None

        self.lock_lost_frames += 1
        self.lost_frame_count = self.lock_lost_frames
        if self.lock_lost_frames >= self.max_lost_window:
            self._reset_tracking_state()
        return None

    def _reset_tracking_state(self):
        self.is_monopoly_locked = False
        self.locked_target_id = None
        self.last_locked_box = None
        self.last_tracked_center = None
        self.last_tracked_target = None
        self.prev_target_center = None
        self.smoothed_target_center = None
        self.prev_error_norm = None
        self.last_anchor_class = None
        self.lock_lost_frames = 0
        self.lost_frame_count = 0

    def _update_tracking_state(self, target, source_row=None):
        if target is None:
            return
        self.last_tracked_center = (float(target.x), float(target.y))
        self.last_tracked_target = target
        if source_row is not None:
            box = source_row[0].detach().float().cpu().numpy()
            self.last_locked_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            self.locked_target_id = self._row_class(source_row)
            self.is_monopoly_locked = True
        self.lock_lost_frames = 0
        self.lost_frame_count = 0

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
        self.reload_runtime_config()
        # ─── 刚性对齐局部 ROI 坐标系中心 ───
        # YOLO sees the center-cropped detection matrix, so distance ranking and
        # residual math must use the ROI's internal center, not the 1080P screen center.
        roi_center_x = float(self.config.getint("Detection window", "detection_window_width", fallback=448)) / 2.0
        roi_center_y = float(self.config.getint("Detection window", "detection_window_height", fallback=448)) / 2.0
        center = torch.tensor([roi_center_x, roi_center_y], device=self.arch)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        weights = torch.ones_like(distances_sq)
        classes_long = torch.round(classes_tensor.float()).long()

        head_classes = self._profile_head_classes()
        body_classes = self._profile_body_classes()
        strict_head_mask = torch.zeros_like(classes_long, dtype=torch.bool)
        for cls_id in head_classes:
            strict_head_mask |= classes_long == int(cls_id)
        class01_mask = torch.zeros_like(classes_long, dtype=torch.bool)
        for cls_id in body_classes:
            class01_mask |= classes_long == int(cls_id)
        if strict_head_mask.any():
            nearest_idx = torch.argmin(distances_sq[strict_head_mask])
            nearest_idx = torch.nonzero(strict_head_mask)[nearest_idx].item()
        elif class01_mask.any():
            nearest_idx = torch.argmin(distances_sq[class01_mask])
            nearest_idx = torch.nonzero(class01_mask)[nearest_idx].item()
        elif cfg.disable_headshot:
            non_head_mask = classes_long != 7
            weights = torch.ones_like(distances_sq)
            weights[classes_long == 7] *= 0.5
            size_factor = boxes_array[:, 2] * boxes_array[:, 3]
            distances_sq = weights * (distances_sq / size_factor)

            if not non_head_mask.any():
                return None
            nearest_idx = torch.argmin(distances_sq[non_head_mask])
            nearest_idx = torch.nonzero(non_head_mask)[nearest_idx].item()
        else:
            head_mask = classes_long == 7
            if head_mask.any():
                nearest_idx = torch.argmin(distances_sq[head_mask])
                nearest_idx = torch.nonzero(head_mask)[nearest_idx].item()
            else:
                nearest_idx = torch.argmin(distances_sq)

        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = int(classes_long[nearest_idx].item())

        target_x, target_y, target_w, target_h = target_data
        x_min = target_x - target_w / 2
        y_min = target_y - target_h / 2
        x_max = target_x + target_w / 2
        y_max = target_y + target_h / 2
        # ─── 刚性对齐 448x448 局部坐标系中心 ───
        # Drop any 960/540 full-screen center usage. YOLO coordinates [0..ROI]
        # now align directly with this local center, so the residual is native 1:1.
        roi_center_x = float(self.config.getint("Detection window", "detection_window_width", fallback=448)) / 2.0
        roi_center_y = float(self.config.getint("Detection window", "detection_window_height", fallback=448)) / 2.0

        measured_center_x = (x_min + x_max) / 2
        if int(target_class) in self._profile_head_classes():
            # Head/keypoint anchors use a high-precision upper reference, never Body/feet baseline.
            measured_center_y = y_min + 0.15 * target_h
        else:
            measured_center_y = (y_min + y_max) / 2
        measured_center = (float(measured_center_x), float(measured_center_y))

        if self.last_anchor_class != int(target_class):
            self.smoothed_target_center = None
            self.prev_target_center = None
            self.prev_error_norm = None
            self.last_anchor_class = int(target_class)

        if self.smoothed_target_center is None:
            self.smoothed_target_center = measured_center
        else:
            alpha = self.anchor_smoothing_alpha
            # Y axis is intentionally more damped; bbox top/bottom flicker is the main vertical jitter source.
            y_alpha = min(alpha, 0.18)
            self.smoothed_target_center = (
                alpha * measured_center[0] + (1.0 - alpha) * self.smoothed_target_center[0],
                y_alpha * measured_center[1] + (1.0 - y_alpha) * self.smoothed_target_center[1],
            )

        target_center_x, target_center_y = self.smoothed_target_center
        current_center = (float(target_center_x), float(target_center_y))
        velocity_x = 0.0
        velocity_y = 0.0
        if self.prev_target_center is not None:
            velocity_x = (current_center[0] - self.prev_target_center[0]) / max(self.dt, 1e-6)
            velocity_y = (current_center[1] - self.prev_target_center[1]) / max(self.dt, 1e-6)
        self.prev_target_center = current_center
        velocity_abs = float((velocity_x ** 2 + velocity_y ** 2) ** 0.5)
        if abs(velocity_x) < self.feedforward_deadzone_pixels:
            velocity_x = 0.0
        if abs(velocity_y) < self.feedforward_y_deadzone_pixels:
            velocity_y = 0.0
        velocity_y = max(-self.max_feedforward_velocity_y, min(self.max_feedforward_velocity_y, velocity_y))
        velocity_abs = float((velocity_x ** 2 + velocity_y ** 2) ** 0.5)
        if velocity_abs > self.max_feedforward_velocity:
            clamp_scale = self.max_feedforward_velocity / max(velocity_abs, 1e-6)
            velocity_x *= clamp_scale
            velocity_y *= clamp_scale
            velocity_abs = self.max_feedforward_velocity

        # Target Bounding Box Containment Gate:
        # If the ROI center/crosshair is already inside the selected target box,
        # use zero spatial error but still allow velocity feed-forward for moving targets.
        is_inside_target = (x_min <= roi_center_x <= x_max) and (y_min <= roi_center_y <= y_max)

        if is_inside_target and int(target_class) not in self._profile_head_classes():
            raw_dx = 0.0
            raw_dy = 0.0
        elif int(target_class) in self._profile_head_classes():
            # Head/keypoint labels are terminal precision anchors: use smoothed upper anchor.
            raw_dx = target_center_x - roi_center_x
            raw_dy = target_center_y - roi_center_y
        else:
            raw_dx, raw_dy = self.dynamic_focus_parser.update_focus_offset(
                [x_min, y_min, x_max, y_max],
                screen_center=(roi_center_x, roi_center_y),
            )

        raw_error_norm = float((float(raw_dx) ** 2 + float(raw_dy) ** 2) ** 0.5)
        prediction_gate = 1.0
        if self.prev_error_norm is not None and raw_error_norm < self.prev_error_norm and raw_error_norm < 60.0:
            # When the residual is already collapsing into the target, brake feed-forward to avoid overshoot.
            prediction_gate = max(0.0, min(1.0, raw_error_norm / 60.0))
        self.prev_error_norm = raw_error_norm

        coordinate_mapping_scale = 1.0
        final_pixel_dx = (float(raw_dx) + velocity_x * self.prediction_factor * prediction_gate) * coordinate_mapping_scale
        final_pixel_dy = (float(raw_dy) + velocity_y * self.prediction_factor_y * prediction_gate) * coordinate_mapping_scale

        focused_x = roi_center_x + final_pixel_dx
        focused_y = roi_center_y + final_pixel_dy

        return Target(focused_x, focused_y, target_w, target_h, target_class, focused=True, velocity_abs=velocity_abs)

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        elif 'cpu' in cfg.AI_device:
            return 'cpu'
        else:
            return f'cuda:{cfg.AI_device}'

frameParser = FrameParser()
