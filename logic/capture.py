import configparser
import cv2
import threading
import queue
import time
import numpy as np

from logic.config_watcher import cfg
from logic.logger import logger


class Capture(threading.Thread):
    DEFAULT_SOURCE_PATH = r"F:\yolo_training\game_test..mp4"
    USB_CAPTURE_INDEX = 1
    USB_CAPTURE_WIDTH = 1920
    USB_CAPTURE_HEIGHT = 1080
    USB_CAPTURE_FPS = 60
    RECONNECT_DELAY_SECONDS = 1.0

    def __init__(self):
        super().__init__()
        self.daemon = True
        self.name = "Capture"

        self.print_startup_messages()

        self.screen_x_center = int(cfg.detection_window_width / 2)
        self.screen_y_center = int(cfg.detection_window_height / 2)
        self.prev_detection_window_width = cfg.detection_window_width
        self.prev_detection_window_height = cfg.detection_window_height
        logger.info(f"[Capture] Detection ROI initialized: {self.prev_detection_window_width}x{self.prev_detection_window_height}")

        self.frame_queue = queue.Queue(maxsize=1)
        self.config = configparser.ConfigParser()
        self.config_last_read = 0.0
        self.config_reload_interval = 0.20
        self.running = True
        self.cap = None
        self.static_frame = None
        self.last_reconnect_attempt = 0.0
        self.source_mode = cfg.source_mode
        self.simulation_mode = cfg.simulation_mode

        self.setup_capture()

    def setup_capture(self):
        mode = self.read_source_mode()
        if mode == "image":
            return self.setup_static_image_capture()
        if mode == "video":
            return self.setup_simulation_capture()
        if mode == "obs":
            return self.setup_obs_capture()
        return self.setup_usb_capture()

    def setup_static_image_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.source_mode = "image"
        self.simulation_mode = True
        image_path = getattr(cfg, "source_path", self.DEFAULT_SOURCE_PATH)
        frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            logger.error(f"[Capture] Static image could not be opened: {image_path}")
            self.static_frame = None
            return False

        self.static_frame = frame
        height, width = frame.shape[:2]
        logger.info(f"[Capture] Static image initialized (path={image_path}, actual={width}x{height})")
        return True

    def setup_simulation_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.static_frame = None
        self.source_mode = "video"
        self.simulation_mode = True
        video_path = getattr(cfg, "source_path", getattr(cfg, "simulation_video_path", self.DEFAULT_SOURCE_PATH))
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error(f"[Capture] Simulation video could not be opened: {video_path}")
            return False

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "[Capture] Simulation video initialized "
            f"(path={video_path}, frames={total_frames}, "
            f"actual={actual_width}x{actual_height}@{actual_fps:.2f})"
        )
        return True

    def setup_obs_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.static_frame = None
        self.source_mode = "obs"
        self.simulation_mode = False
        # Force DirectShow to bypass OpenCV's obsensor auto-probe path.
        camera_index = self.read_obs_camera_index()
        obs_idx = int(camera_index)
        self.cap = cv2.VideoCapture(obs_idx, cv2.CAP_DSHOW)

        # ─── 强行握手 1080P 核心输入总线分辨率 ───
        # 刚性指定驱动层必须以 1920x1080 满血分辨率输出数据帧，避免 OpenCV 默认 640x480 下采样。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 保持低延迟编码格式。
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, self.USB_CAPTURE_FPS)

        if not self.cap.isOpened():
            logger.error(f"[Capture] OBS virtual camera index {camera_index} could not be opened")
            return False

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "[Capture] OBS virtual bus initialized "
            f"(index={camera_index}, fourcc=MJPG, actual={actual_width}x{actual_height}@{actual_fps:.2f})"
        )
        return True

    def setup_usb_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.static_frame = None
        self.source_mode = "hardware"
        self.simulation_mode = False
        self.cap = cv2.VideoCapture(self.USB_CAPTURE_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.USB_CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.USB_CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.USB_CAPTURE_FPS)

        if not self.cap.isOpened():
            logger.error(f"[Capture] USB capture card index {self.USB_CAPTURE_INDEX} could not be opened")
            return False

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "[Capture] USB capture card initialized "
            f"(index={self.USB_CAPTURE_INDEX}, requested={self.USB_CAPTURE_WIDTH}x"
            f"{self.USB_CAPTURE_HEIGHT}@{self.USB_CAPTURE_FPS}, actual={actual_width}x"
            f"{actual_height}@{actual_fps:.2f})"
        )
        return True

    def run(self):
        try:
            while self.running:
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame, block=False)
        finally:
            self.release_capture()

    def capture_frame(self):
        self.reload_source_mode_if_needed()

        if self.source_mode == "image":
            if self.static_frame is None:
                self.reconnect_capture()
                return None
            return self.prepare_frame(self.static_frame.copy())

        if self.cap is None or not self.cap.isOpened():
            self.reconnect_capture()
            return None

        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.error(f"[Capture] frame read exception: {e}")
            self.reconnect_capture(force=True)
            return None

        if ret and frame is not None:
            return self.prepare_frame(frame)

        if self.source_mode == "video":
            logger.info("[Capture] Simulation video reached EOF; rewinding to frame 0")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return self.prepare_frame(frame)
            logger.warning("[Capture] Simulation rewind failed, reconnecting video source")
            self.reconnect_capture(force=True)
            return None

        logger.warning(f"[Capture] {self.source_mode} frame read failed, reconnecting capture source")
        self.reconnect_capture(force=True)
        return None

    def prepare_frame(self, frame):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        target_w, target_h = self.read_detection_window_size()
        self.update_detection_window(target_w, target_h)
        return frame

    def read_config(self, force=False):
        now = time.monotonic()
        if (not force) and (now - self.config_last_read < self.config_reload_interval):
            return self.config
        self.config.clear()
        self.config.read("config.ini", encoding="utf-8")
        self.config_last_read = now
        return self.config

    def read_source_mode(self):
        try:
            mode = self.read_config().get("Capture Methods", "source_mode", fallback=getattr(cfg, "source_mode", "video")).strip().lower()
            return mode if mode in {"hardware", "video", "image", "obs"} else "video"
        except Exception:
            return getattr(cfg, "source_mode", "video")

    def read_obs_camera_index(self):
        try:
            return self.read_config().getint("Capture Methods", "obs_camera_index", fallback=getattr(cfg, "obs_camera_index", 1))
        except Exception:
            return getattr(cfg, "obs_camera_index", 1)

    def reload_source_mode_if_needed(self):
        current_mode = self.read_source_mode()
        if current_mode == self.source_mode:
            return

        logger.info(f"[Capture] source_mode reloaded: {self.source_mode} -> {current_mode}")
        cfg.source_mode = current_mode
        cfg.simulation_mode = current_mode in {"video", "image"}
        self.release_capture()
        self.source_mode = current_mode
        self.setup_capture()

    def read_detection_window_size(self):
        """Hot-read the central ROI size from config.ini for per-frame crop changes."""
        try:
            parser = self.read_config()
            crop_width = parser.getint("Detection window", "detection_window_width", fallback=self.prev_detection_window_width)
            crop_height = parser.getint("Detection window", "detection_window_height", fallback=self.prev_detection_window_height)
            return max(32, crop_width), max(32, crop_height)
        except Exception:
            return self.prev_detection_window_width, self.prev_detection_window_height

    def update_detection_window(self, crop_width, crop_height):
        if crop_width == self.prev_detection_window_width and crop_height == self.prev_detection_window_height:
            return

        self.screen_x_center = int(crop_width / 2)
        self.screen_y_center = int(crop_height / 2)
        cfg.detection_window_width = crop_width
        cfg.detection_window_height = crop_height
        self.prev_detection_window_width = crop_width
        self.prev_detection_window_height = crop_height
        logger.info(f"[Capture] Detection ROI reloaded: {crop_width}x{crop_height} (center={self.screen_x_center},{self.screen_y_center})")

    def reconnect_capture(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_reconnect_attempt < self.RECONNECT_DELAY_SECONDS:
            return

        self.last_reconnect_attempt = now
        self.release_capture()
        time.sleep(0.2)
        self.setup_capture()

    def release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.static_frame = None

    def get_new_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def restart(self):
        if (
            self.prev_detection_window_height != cfg.detection_window_height or
            self.prev_detection_window_width != cfg.detection_window_width
        ):
            self.screen_x_center = int(cfg.detection_window_width / 2)
            self.screen_y_center = int(cfg.detection_window_height / 2)
            self.prev_detection_window_width = cfg.detection_window_width
            self.prev_detection_window_height = cfg.detection_window_height

        self.reconnect_capture(force=True)
        logger.info("[Capture] capture source reloaded")

    def print_startup_messages(self):
        version = 0
        try:
            with open("./version", "r") as f:
                version = f.readline().split("=")[1].strip()
        except FileNotFoundError:
            logger.info("(version file is not found)")
        except Exception as e:
            logger.info(f"Error with read version file: {str(e)}")

        logger.info(f"""
Sunone Aimbot is started! (Version {version})
Hotkeys:
[{cfg.hotkey_targeting}] - Aiming at the target
[{cfg.hotkey_exit}] - EXIT
[{cfg.hotkey_pause}] - PAUSE AIM
[{cfg.hotkey_reload_config}] - Reload config
""")

    def convert_to_circle(self, image):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (width // 2, height // 2), (width // 2, height // 2), 0, 0, 360, 255, -1)
        return cv2.bitwise_and(image, cv2.merge([mask, mask, mask]))

    def Quit(self):
        self.running = False
        self.release_capture()
        self.join()


capture = Capture()
capture.start()
