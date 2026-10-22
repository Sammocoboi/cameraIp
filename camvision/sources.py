"""Источники видео: веб-камера, файл, IP-камера (RTSP/HTTP), картинка."""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# RTSP по TCP надёжнее по Wi-Fi/локальной сети, чем UDP по умолчанию.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def looks_like_stream(source: str) -> bool:
    low = source.lower()
    return low.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "rtmp://"))


def is_image_source(source: str) -> bool:
    if looks_like_stream(source):
        return Path(source.split("?")[0]).suffix.lower() in IMAGE_SUFFIXES
    return Path(source).suffix.lower() in IMAGE_SUFFIXES


class VideoSource:
    """Единый интерфейс чтения кадров."""

    def __init__(
        self,
        source: int | str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = None,
        loop: bool = False,
        reconnect: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps_request = fps
        self.fourcc = fourcc
        self.loop = loop
        self.reconnect = reconnect
        self.cap: cv2.VideoCapture | None = None
        self.is_live = isinstance(source, int) or (
            isinstance(source, str) and looks_like_stream(source)
        )
        self._last_frame: np.ndarray | None = None
        self._reconnect_attempts = 0

    # ------------------------------------------------------------------
    def open(self) -> None:
        if isinstance(self.source, int):
            self.cap = self._open_camera(self.source)
        else:
            self.cap = self._open_path(self.source)
        if self.cap is None or not self.cap.isOpened():
            self.release()
            if isinstance(self.source, int):
                raise RuntimeError(
                    f"Камера {self.source} не открылась. Проверьте, что она не занята "
                    "другим приложением, и разрешён доступ к камере в настройках Windows."
                )
            raise RuntimeError(f"Не удалось открыть источник: {self.source}")

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps_request:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps_request))
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _open_camera(self, index: int) -> cv2.VideoCapture:
        backends = []
        if sys.platform == "win32":
            backends.append(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
        backends.append(cv2.CAP_ANY)
        cap = None
        for backend in backends:
            try:
                candidate = cv2.VideoCapture(index, backend)
            except Exception:
                continue
            if candidate.isOpened():
                ok, frame = candidate.read()
                if ok and frame is not None:
                    return candidate
                candidate.release()
            else:
                candidate.release()
        return cap or cv2.VideoCapture(index)

    def _open_path(self, path: str) -> cv2.VideoCapture:
        if looks_like_stream(path):
            return cv2.VideoCapture(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return cv2.VideoCapture(path)

    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._last_frame = frame
            self._reconnect_attempts = 0
            return True, frame

        if self.is_live and self.reconnect and self._reconnect_attempts < 3:
            self._reconnect_attempts += 1
            time.sleep(0.4)
            self.reopen()
            if self.cap is not None:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self._last_frame = frame
                    return True, frame

        if self.loop and not self.is_live:
            self.reopen()
            if self.cap is not None:
                return self.cap.read()
        return False, None

    def reopen(self) -> None:
        self.release()
        try:
            self.open()
        except Exception:
            self.cap = None

    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        if self.cap is None:
            return 20.0
        value = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if value <= 1.0 or value > 240.0:
            return 20.0 if self.is_live else 25.0
        return value

    @property
    def frame_size(self) -> tuple[int, int]:
        if self.cap is None:
            return 0, 0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            if self._last_frame is not None:
                h, w = self._last_frame.shape[:2]
        return w, h

    def describe(self) -> str:
        if isinstance(self.source, int):
            return f"камера #{self.source}"
        return str(self.source)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CameraVision/1.0 (+https://github.com/)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def load_image(source: str) -> np.ndarray:
    """Читает картинку с диска или по http(s)-ссылке."""
    if looks_like_stream(source):
        raw = np.frombuffer(_download(source), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {source}")
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {source}")
    return image


def probe_cameras(max_index: int = 6, timeout_s: float = 2.0) -> list[tuple[int, int, int]]:
    """Пробует открыть камеры 0..max_index-1 и возвращает (индекс, ширина, высота)."""
    found: list[tuple[int, int, int]] = []
    for index in range(max_index):
        cap = None
        started = time.perf_counter()
        try:
            backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else []
            backends.append(cv2.CAP_ANY)
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        found.append((index, w, h))
                        break
                cap.release()
                cap = None
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
        if time.perf_counter() - started > timeout_s and not found:
            continue
    return found
