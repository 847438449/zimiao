import win32con, win32api
import time
import math
import os
import configparser
import queue
import json
import random
import socket
import sys
import supervision as sv

from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.buttons import Buttons
from logic.logger import logger

if cfg.mouse_rzr:
    from logic.rzctl import RZCONTROL

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

class MouseThread:
    def __init__(self):
        self.initialize_parameters()
        self.setup_hardware()

    def initialize_parameters(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        # Prediction is centralized in frame_parser.py. Keep the legacy mouse-layer
        # predictor hard-disabled to avoid double feed-forward / overshoot.
        self.disable_prediction = True
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.prev_distance = None
        self.speed_correction_factor = 0.1
        self.bScope = False
        self.arch = self.get_arch()
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        self.udp_output = cfg.udp_output
        self.udp_host = cfg.udp_host
        self.udp_port = cfg.udp_port
        self.udp_send_when_key_pressed_only = cfg.udp_send_when_key_pressed_only
        self.udp_send_json = cfg.udp_send_json
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_output else None
        self.current_aim_mode = "UDP" if self.udp_output else "MOVE"
        self.runtime_config = configparser.ConfigParser()
        self.runtime_config_last_read = 0.0
        self.runtime_config_reload_interval = 0.05
        self.local_residual_x = 0.0
        self.local_residual_y = 0.0
        self.subpixel_deadzone_pixels = 0.45
        self.subpixel_y_deadzone_pixels = 0.75
        self.local_output_gain = 0.85
        self.last_stream_log_time = 0.0
        self.stream_log_interval = 0.10
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        if 'cpu' in cfg.AI_device:
            return 'cpu'
        return f'cuda:{cfg.AI_device}'

    def setup_hardware(self):
        if cfg.mouse_ghub:
            from logic.ghub import gHub
            self.ghub = gHub

        if cfg.mouse_rzr:
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rzctl.dll")
            self.rzr = RZCONTROL(dll_path)
            if not self.rzr.init():
                logger.error("Failed to initialize rzctl")

    def process_data(self, data):
        target_velocity_abs = 0.0
        if isinstance(data, sv.Detections):
            target_x, target_y = data.xyxy.mean(axis=1)
            target_w, target_h = data.xyxy[:, 2] - data.xyxy[:, 0], data.xyxy[:, 3] - data.xyxy[:, 1]
            target_cls = data.class_id[0] if data.class_id.size > 0 else None
        else:
            if len(data) >= 6:
                target_x, target_y, target_w, target_h, target_cls, target_velocity_abs = data
            else:
                target_x, target_y, target_w, target_h, target_cls = data

        self.visualize_target(target_x, target_y, target_cls)
        self.bScope = self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bScope_multiplier) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        if not self.disable_prediction:
            # Deprecated safeguard: normal runtime keeps this branch disabled.
            current_time = time.time()
            if not isinstance(data, sv.Detections):
                target_x, target_y = self.predict_target_position(target_x, target_y, current_time)
            self.visualize_prediction(target_x, target_y, target_cls)

        self.refresh_geometry_from_config()
        dx = target_x - self.center_x
        dy = target_y - self.center_y
        shooting_state = self.get_shooting_key_state()

        self.visualize_history(target_x, target_y)
        self.enqueue_shooting_state(self.bScope, shooting_state)

        if self.is_inside_error_deadzone(dx, dy):
            self.reset_ema_filter()
            dx, dy = 0.0, 0.0
            decay_gain = 0.0
            scale_factor = self.read_scale_adjustment_factor()
            random_multiplier = 1.0
        else:
            dx, dy = self.apply_ema_filter(dx, dy, target_velocity_abs)
            dx, dy, scale_factor = self.apply_error_adaptive_gain(dx, dy)
            dx, dy, random_multiplier = self.apply_udp_random_multiplier(dx, dy, target_velocity_abs)
            dx, dy, decay_gain = self.apply_capture_zone_gain_decay(dx, dy, target_velocity_abs)
            dx, dy = self.apply_saturation_clamp(dx, dy)

        source_mode = self.read_current_source_mode()
        self.dispatch_control_pulse(
            dx,
            dy,
            target_x,
            target_y,
            target_w,
            target_h,
            target_cls,
            shooting_state,
            source_mode,
            random_multiplier,
            scale_factor,
            decay_gain,
        )

    def enqueue_shooting_state(self, b_scope, shooting_state):
        """Publish shooting state without blocking the high-frequency movement loop."""
        try:
            if shooting.queue.full():
                try:
                    shooting.queue.get_nowait()
                except queue.Empty:
                    pass
            shooting.queue.put_nowait((b_scope, shooting_state))
        except queue.Full:
            pass

    def read_runtime_config(self, force=False):
        now = time.monotonic()
        if (not force) and (now - self.runtime_config_last_read < self.runtime_config_reload_interval):
            return self.runtime_config
        self.runtime_config.clear()
        self.runtime_config.read("config.ini", encoding="utf-8")
        self.runtime_config_last_read = now
        return self.runtime_config

    def read_current_source_mode(self):
        try:
            current_source_mode = self.read_runtime_config().get("Capture Methods", "source_mode", fallback=getattr(cfg, "source_mode", "obs"))
            return current_source_mode.strip().lower()
        except Exception:
            return getattr(cfg, "source_mode", "obs")

    def read_ema_alpha(self):
        try:
            alpha = self.read_runtime_config().getfloat("Control_Filter", "ema_alpha", fallback=getattr(cfg, "ema_alpha", 0.35))
        except Exception:
            alpha = getattr(cfg, "ema_alpha", 0.35)
        return max(0.0, min(1.0, float(alpha)))

    def read_error_deadzone_pixels(self):
        try:
            return max(0.0, self.read_runtime_config().getfloat("Control_Filter", "error_deadzone_pixels", fallback=2.0))
        except Exception:
            return max(0.0, float(getattr(cfg, "error_deadzone_pixels", 2.0)))

    def read_capture_zone_radius(self):
        try:
            return max(1e-6, self.read_runtime_config().getfloat("Control_Filter", "capture_zone_radius", fallback=20.0))
        except Exception:
            return max(1e-6, float(getattr(cfg, "capture_zone_radius", 20.0)))

    def is_inside_error_deadzone(self, dx, dy):
        epsilon = self.read_error_deadzone_pixels()
        return abs(float(dx)) <= epsilon and abs(float(dy)) <= epsilon

    def reset_ema_filter(self):
        self.ema_dx, self.ema_dy = 0.0, 0.0

    def adaptive_ema_alpha(self, dx, dy, velocity_abs=0.0):
        default_alpha = self.read_ema_alpha()
        deadzone = self.read_error_deadzone_pixels()
        capture_zone = max(deadzone + 1e-6, self.read_capture_zone_radius())
        error_norm = math.sqrt(float(dx) ** 2 + float(dy) ** 2)

        if error_norm > 30.0 or float(velocity_abs) > 3.0:
            error_boost = max(0.0, min(1.0, (error_norm - 30.0) / 70.0))
            velocity_boost = max(0.0, min(1.0, (float(velocity_abs) - 3.0) / 12.0))
            dynamic_alpha = 0.70 + 0.25 * max(error_boost, velocity_boost)
            return max(float(default_alpha), min(0.95, dynamic_alpha))

        min_alpha = min(default_alpha, 0.12)
        normalized = max(0.0, min(1.0, (error_norm - deadzone) / (capture_zone - deadzone)))
        return min_alpha + (default_alpha - min_alpha) * normalized

    def apply_ema_filter(self, dx, dy, velocity_abs=0.0):
        """Apply deviation-sensitive first-order EMA before scaling."""
        alpha = self.adaptive_ema_alpha(dx, dy, velocity_abs)
        if not hasattr(self, "ema_dx"):
            self.ema_dx, self.ema_dy = dx, dy
        else:
            self.ema_dx = alpha * dx + (1 - alpha) * self.ema_dx
            self.ema_dy = alpha * dy + (1 - alpha) * self.ema_dy
        return self.ema_dx, self.ema_dy

    def apply_capture_zone_gain_decay(self, dx, dy, velocity_abs=0.0):
        """Apply nonlinear gain decay unless target velocity needs full tracking force."""
        if float(velocity_abs) > 3.0:
            return dx, dy, 1.0

        capture_zone = self.read_capture_zone_radius()
        error_norm = math.sqrt(float(dx) ** 2 + float(dy) ** 2)
        if error_norm >= capture_zone:
            return dx, dy, 1.0

        min_gain = 0.15
        normalized = max(0.0, min(1.0, error_norm / capture_zone))
        curve = (1.0 - math.exp(-3.0 * normalized)) / (1.0 - math.exp(-3.0))
        decay_gain = min_gain + (1.0 - min_gain) * curve
        return dx * decay_gain, dy * decay_gain, decay_gain

    def apply_udp_random_multiplier(self, dx, dy, velocity_abs=0.0):
        """Apply bounded stochastic damping; bypass under high-bandwidth tracking demand."""
        error_norm = math.sqrt(float(dx) ** 2 + float(dy) ** 2)
        if error_norm > 30.0 or float(velocity_abs) > 3.0:
            return dx, dy, 1.0

        min_mult = float(getattr(cfg, "mouse_min_speed_multiplier", self.min_speed_multiplier))
        max_mult = float(getattr(cfg, "mouse_max_speed_multiplier", self.max_speed_multiplier))
        min_mult = max(0.80, min(1.0, min_mult))
        max_mult = max(0.80, min(1.0, max_mult))
        if min_mult > max_mult:
            min_mult, max_mult = max_mult, min_mult

        current_random_multiplier = random.uniform(min_mult, max_mult)
        return dx * current_random_multiplier, dy * current_random_multiplier, current_random_multiplier

    def read_max_control_step_pixels(self):
        try:
            return max(1.0, self.read_runtime_config().getfloat("Control_Filter", "max_control_step_pixels", fallback=32.0))
        except Exception:
            return max(1.0, float(getattr(cfg, "max_control_step_pixels", 32.0)))

    def apply_saturation_clamp(self, dx, dy):
        """Critically damped vector clamp: fast acquisition without crossing the residual."""
        base_step = self.read_max_control_step_pixels()
        error_norm = math.sqrt(float(dx) ** 2 + float(dy) ** 2)
        if error_norm <= 1e-6:
            return 0.0, 0.0

        if error_norm > 30.0:
            release = min(18.0, math.log1p((error_norm - 30.0) / 25.0) * 12.0)
            max_step = base_step + release
        else:
            max_step = base_step

        # Brake as the residual shrinks: never command a full jump across the target.
        if error_norm < 18.0:
            residual_fraction_limit = max(1.0, error_norm * 0.45)
        elif error_norm < 45.0:
            residual_fraction_limit = max(3.0, error_norm * 0.62)
        else:
            residual_fraction_limit = max(8.0, error_norm * 0.72)

        max_step = min(max_step, residual_fraction_limit)
        if error_norm <= max_step:
            return float(dx), float(dy)

        scale = max_step / error_norm
        return float(dx) * scale, float(dy) * scale

    def refresh_geometry_from_config(self):
        """Hot-read ROI dimensions so center math tracks 320/448 UI switches."""
        try:
            parser = configparser.ConfigParser()
            parser.read("config.ini", encoding="utf-8")
            screen_width = parser.getint("Detection window", "detection_window_width", fallback=int(self.screen_width))
            screen_height = parser.getint("Detection window", "detection_window_height", fallback=int(self.screen_height))
        except Exception:
            return

        if screen_width == self.screen_width and screen_height == self.screen_height:
            return

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2
        self.section_size_x = self.screen_width / 100
        self.section_size_y = self.screen_height / 100
        logger.info(f"[Mouse Stream] Geometry reloaded: ROI={screen_width}x{screen_height} center=({self.center_x:.1f},{self.center_y:.1f})")

    def read_scale_adjustment_factor(self):
        """Read the latest abstract gain factor from config.ini for hot UI switching."""
        try:
            parser = self.read_runtime_config()
            return parser.getfloat("Control_Filter", "scale_adjustment_factor", fallback=1.0)
        except Exception:
            return float(getattr(cfg, "scale_adjustment_factor", 1.0))

    def apply_resolution_scale(self, dx, dy):
        """Apply profile-level gain compensation to smoothed deltas."""
        scale_factor = self.read_scale_adjustment_factor()
        scaled_dx = dx * scale_factor
        scaled_dy = dy * scale_factor
        return scaled_dx, scaled_dy, scale_factor

    def apply_error_adaptive_gain(self, dx, dy):
        """Nonlinear error-adaptive gain: high bandwidth for large residuals, damping near lock."""
        base_scale = self.read_scale_adjustment_factor()
        error_norm = math.sqrt(float(dx) ** 2 + float(dy) ** 2)

        if error_norm > 30.0:
            # Moderate boost for faster acquisition while keeping overshoot bounded.
            normalized = max(0.0, min(1.0, (error_norm - 30.0) / 70.0))
            boost_scale = 1.12 + 0.18 * normalized
            scale_factor = max(float(base_scale), boost_scale)
        else:
            # Precision adsorption zone: return to the configured damping scale.
            scale_factor = float(base_scale)

        return dx * scale_factor, dy * scale_factor, scale_factor

    def log_mouse_stream(self, dx, dy, target_cls, random_multiplier=None, scale_factor=None, decay_gain=None):
        """Print a throttled mouse stream sample; never flush-log every control pulse."""
        now = time.monotonic()
        if now - self.last_stream_log_time < self.stream_log_interval:
            return
        self.last_stream_log_time = now

        if target_cls is None:
            cls_str = "❔ UNKNOWN"
        else:
            target_cls = int(target_cls)
            if target_cls == 0:
                cls_str = "🧠 HEAD"
            elif target_cls == 1:
                cls_str = "👕 BODY"
            else:
                cls_str = f"CLS {target_cls}"

        rand_part = "" if random_multiplier is None else f" | Rand: {float(random_multiplier):.3f}"
        scale_part = "" if scale_factor is None else f" | Scale: {float(scale_factor):.4g}"
        decay_part = "" if decay_gain is None else f" | Decay: {float(decay_gain):.3f}"
        message = (
            f"[Mouse Stream] Target: {cls_str:<10} | "
            f"dx: {float(dx):+6.1f} | dy: {float(dy):+6.1f} | "
            f"Mode: {self.current_aim_mode:<4}"
            f"{rand_part}"
            f"{scale_part}"
            f"{decay_part}"
        )
        try:
            print(message, flush=True)
        except UnicodeEncodeError:
            print(message.encode("utf-8", errors="replace").decode("utf-8"), flush=True)

    def predict_target_position(self, target_x, target_y, current_time):
        # First target
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_x = target_x
            self.prev_y = target_y
            self.prev_velocity_x = 0
            self.prev_velocity_y = 0
            return target_x, target_y

        # Next target?
        max_jump = max(self.screen_width, self.screen_height) * 0.3 # 30%
        if abs(target_x - self.prev_x) > max_jump or abs(target_y - self.prev_y) > max_jump:
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_velocity_x = 0
            self.prev_velocity_y = 0
            self.prev_time = current_time
            return target_x, target_y

        delta_time = current_time - self.prev_time

        if delta_time == 0:
            delta_time = 1e-6

        velocity_x = (target_x - self.prev_x) / delta_time
        velocity_y = (target_y - self.prev_y) / delta_time
        acceleration_x = (velocity_x - self.prev_velocity_x) / delta_time
        acceleration_y = (velocity_y - self.prev_velocity_y) / delta_time

        prediction_interval = delta_time * self.prediction_interval
        current_distance = math.sqrt((target_x - self.prev_x)**2 + (target_y - self.prev_y)**2)
        proximity_factor = max(0.1, min(1, 1 / (current_distance + 1)))

        speed_correction = 1 + (abs(current_distance - (self.prev_distance or 0)) / self.max_distance) * self.speed_correction_factor if self.prev_distance is not None else .0001

        predicted_x = target_x + velocity_x * prediction_interval * proximity_factor * speed_correction + 0.5 * acceleration_x * (prediction_interval ** 2) * proximity_factor * speed_correction
        predicted_y = target_y + velocity_y * prediction_interval * proximity_factor * speed_correction + 0.5 * acceleration_y * (prediction_interval ** 2) * proximity_factor * speed_correction

        self.prev_x, self.prev_y = target_x, target_y
        self.prev_velocity_x, self.prev_velocity_y = velocity_x, velocity_y
        self.prev_time = current_time
        self.prev_distance = current_distance

        return predicted_x, predicted_y

    def calculate_speed_multiplier(self, target_x, target_y, distance):
        if any(map(math.isnan, (target_x, target_y))) or self.section_size_x == 0:
            return self.min_speed_multiplier

        normalized_distance = min(distance / self.max_distance, 1)
        base_speed = self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * (1 - normalized_distance)

        if self.section_size_x == 0:
            return self.min_speed_multiplier

        target_x_section = int((target_x - self.center_x + self.screen_width / 2) / self.section_size_x)
        target_y_section = int((target_y - self.center_y + self.screen_height / 2) / self.section_size_y)

        distance_from_center = max(abs(50 - target_x_section), abs(50 - target_y_section))

        if distance_from_center == 0:
            return 1
        elif 5 <= distance_from_center <= 10:
            return self.max_speed_multiplier
        else:
            speed_reduction = min(distance_from_center - 10, 45) / 100.0
            speed_multiplier = base_speed * (1 - speed_reduction)

        if self.prev_distance is not None:
            speed_adjustment = 1 + (abs(distance - self.prev_distance) / self.max_distance) * self.speed_correction_factor
            return speed_multiplier * speed_adjustment

        return speed_multiplier

    def DEPRECATED_calc_movement(self, target_x, target_y, target_cls):
        raise NotImplementedError("calc_movement is deprecated; direct 1:1 pixel servo path is the only supported output path.")
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        distance = math.sqrt(offset_x**2 + offset_y**2)
        speed_multiplier = self.calculate_speed_multiplier(target_x, target_y, distance)

        degrees_per_pixel_x = self.fov_x / self.screen_width
        degrees_per_pixel_y = self.fov_y / self.screen_height

        mouse_move_x = offset_x * degrees_per_pixel_x
        mouse_move_y = offset_y * degrees_per_pixel_y

        # Apply smoothing
        alpha = 0.85
        if not hasattr(self, 'last_move_x'):
            self.last_move_x, self.last_move_y = 0, 0

        move_x = alpha * mouse_move_x + (1 - alpha) * self.last_move_x
        move_y = alpha * mouse_move_y + (1 - alpha) * self.last_move_y

        self.last_move_x, self.last_move_y = move_x, move_y

        move_x = (move_x / 360) * (self.dpi * (1 / self.mouse_sensitivity)) * speed_multiplier
        move_y = (move_y / 360) * (self.dpi * (1 / self.mouse_sensitivity)) * speed_multiplier

        return move_x, move_y

    def ensure_udp_socket(self):
        if self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def quantize_local_delta(self, dx, dy):
        """Accumulate sub-pixel residuals, but leak the integrator near zero to prevent jitter."""
        if abs(float(dx)) < self.subpixel_deadzone_pixels and abs(float(dy)) < self.subpixel_y_deadzone_pixels:
            self.local_residual_x *= 0.50
            self.local_residual_y *= 0.25
            if abs(self.local_residual_x) < self.subpixel_deadzone_pixels:
                self.local_residual_x = 0.0
            if abs(self.local_residual_y) < self.subpixel_y_deadzone_pixels:
                self.local_residual_y = 0.0
            return 0, 0

        if abs(float(dy)) < self.subpixel_y_deadzone_pixels:
            self.local_residual_y *= 0.25
            dy = 0.0

        total_x = float(dx) + self.local_residual_x
        total_y = float(dy) + self.local_residual_y
        move_x = int(round(total_x))
        move_y = int(round(total_y))
        self.local_residual_x = total_x - move_x
        self.local_residual_y = total_y - move_y
        return move_x, move_y

    def local_move(self, dx, dy):
        """
        Send relative mouse movement through the Windows input bus.

        :param dx: final horizontal pixel delta after filtering/scaling/randomization
        :param dy: final vertical pixel delta after filtering/scaling/randomization
        """
        try:
            gain = self.read_runtime_config().getfloat("Control_Filter", "local_output_gain", fallback=self.local_output_gain)
            gain = max(0.10, min(1.0, float(gain)))
            move_x, move_y = self.quantize_local_delta(float(dx) * gain, float(dy) * gain)
            if move_x == 0 and move_y == 0:
                return
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        except Exception as e:
            print(f"[Mouse Driver Error] Local mouse injection failed: {e}", flush=True)

    def dispatch_control_pulse(
        self,
        dx,
        dy,
        target_x,
        target_y,
        target_w,
        target_h,
        target_cls,
        shooting_state,
        source_mode,
        random_multiplier,
        scale_factor,
        decay_gain,
    ):
        if source_mode == "obs":
            self.current_aim_mode = "LOCALAPI"
            self.local_move(dx, dy)
        else:
            self.current_aim_mode = "KMBOX_UDP"
            self.send_udp_packet(dx, dy, target_x, target_y, target_w, target_h, target_cls, shooting_state)

        self.log_mouse_stream(
            dx,
            dy,
            target_cls,
            random_multiplier=random_multiplier,
            scale_factor=scale_factor,
            decay_gain=decay_gain,
        )

    def send_udp_packet(self, dx, dy, target_x=None, target_y=None, target_w=None, target_h=None, target_cls=None, shooting_state=False):
        """Send the final control vector through the network packet bus."""
        target_x = 0.0 if target_x is None else target_x
        target_y = 0.0 if target_y is None else target_y
        target_w = 0.0 if target_w is None else target_w
        target_h = 0.0 if target_h is None else target_h
        self.send_udp_offset(dx, dy, target_x, target_y, target_w, target_h, target_cls, shooting_state)

    def send_udp_offset(self, dx, dy, target_x, target_y, target_w, target_h, target_cls, shooting_state):
        self.ensure_udp_socket()

        if self.udp_send_when_key_pressed_only and not shooting_state:
            return

        if self.udp_send_json:
            payload = {
                "dx": float(dx),
                "dy": float(dy),
                "target_x": float(target_x),
                "target_y": float(target_y),
                "target_w": float(target_w),
                "target_h": float(target_h),
                "target_cls": None if target_cls is None else int(target_cls),
                "b_scope": bool(self.bScope),
                "shooting_key": bool(shooting_state),
                "timestamp": time.time(),
            }
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        else:
            message = f"{float(dx):.3f},{float(dy):.3f}".encode("ascii")

        try:
            self.udp_socket.sendto(message, (self.udp_host, self.udp_port))
        except OSError as e:
            logger.error(f"UDP send failed: {e}")

    def move_mouse(self, x, y, shooting_state=None):
        if x == 0 and y == 0:
            return

        if shooting_state is None:
            shooting_state = self.get_shooting_key_state()

        if shooting_state or cfg.mouse_auto_aim:
            if not cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
            elif cfg.mouse_ghub:
                self.ghub.mouse_xy(int(x), int(y))
            elif cfg.arduino_move:
                arduino.move(int(x), int(y))
            elif cfg.mouse_rzr:
                self.rzr.mouse_move(int(x), int(y), True)

    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            key_code = Buttons.KEY_CODES.get(key_name.strip())
            if key_code and (win32api.GetKeyState(key_code) if cfg.mouse_lock_target else win32api.GetAsyncKeyState(key_code)) < 0:
                return True
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        reduced_w, reduced_h = target_w * reduction_factor / 2, target_h * reduction_factor / 2
        x1, x2, y1, y2 = target_x - reduced_w, target_x + reduced_w, target_y - reduced_h, target_y + reduced_h
        bScope = self.center_x > x1 and self.center_x < x2 and self.center_y > y1 and self.center_y < y2

        if cfg.show_window and cfg.show_bScope_box:
            visuals.draw_bScope(x1, x2, y1, y2, bScope)

        return bScope

    def update_settings(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = True
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.udp_output = cfg.udp_output
        self.udp_host = cfg.udp_host
        self.udp_port = cfg.udp_port
        self.udp_send_when_key_pressed_only = cfg.udp_send_when_key_pressed_only
        self.udp_send_json = cfg.udp_send_json
        if self.udp_output and self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif not self.udp_output and self.udp_socket is not None:
            self.udp_socket.close()
            self.udp_socket = None

    def visualize_target(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.show_target_line):
            visuals.draw_target_line(target_x, target_y, target_cls)

    def visualize_prediction(self, target_x, target_y, target_cls):
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.show_target_prediction_line):
            visuals.draw_predicted_position(target_x, target_y, target_cls)

    def visualize_history(self, target_x, target_y):
        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(target_x, target_y)

mouse = MouseThread()
