"""Поиск лиц и подсказки по человеку (цвет одежды, положение, дистанция)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .colors import region_color
from .config import MODELS_DIR

log = logging.getLogger("camvision")

YUNET_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_NAME = "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """YuNet (OpenCV FaceDetectorYN) или каскад Хаара как запасной вариант.

    YuNet точнее и лучше работает в профиль; весит ~230 КБ и скачивается
    один раз в models/. В OpenCV 4.x есть оба варианта, в OpenCV 5 — только YuNet.
    """

    def __init__(
        self,
        *,
        min_size: int = 24,
        scale: float = 0.6,
        max_faces: int = 12,
        model_path: str | Path | None = None,
        score_threshold: float = 0.6,
    ) -> None:
        self.scale = float(scale)
        self.min_size = int(min_size)
        self.max_faces = int(max_faces)
        self.score_threshold = float(score_threshold)
        self.error = ""
        self.backend = "нет"
        self._net = None
        self._cascade = None
        self._model_path = Path(model_path) if model_path else MODELS_DIR / YUNET_NAME

        if hasattr(cv2, "FaceDetectorYN_create"):
            self._init_yunet()
        elif hasattr(cv2, "CascadeClassifier"):
            self._init_cascade()
        else:
            self.error = "OpenCV без детектора лиц"

    # ------------------------------------------------------------------
    def _init_yunet(self) -> None:
        try:
            self._ensure_model()
            self._net = cv2.FaceDetectorYN_create(
                str(self._model_path),
                "",
                (320, 320),
                self.score_threshold,
                0.3,
                5000,
            )
            self.backend = "YuNet"
        except Exception as exc:
            self._net = None
            self.error = f"YuNet недоступен: {exc}"
            log.warning(self.error)
            if hasattr(cv2, "CascadeClassifier"):
                self._init_cascade()

    def _ensure_model(self) -> None:
        if self._model_path.exists():
            return
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Скачиваю модель лиц YuNet (~230 КБ) ...")
        request = urllib.request.Request(
            YUNET_URL, headers={"User-Agent": "CameraVision/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = response.read()
        if len(data) < 10_000:
            raise RuntimeError("сервер вернул слишком маленький файл")
        tmp = self._model_path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(self._model_path)

    def _init_cascade(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            self._cascade = cascade
            self.backend = "Haar"

    # ------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self._net is not None or self._cascade is not None

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if frame_bgr.size == 0 or not self.ok:
            return []
        if self._net is not None:
            return self._detect_yunet(frame_bgr)
        return self._detect_cascade(frame_bgr)

    def _detect_yunet(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        sw = max(32, int(w * self.scale))
        sh = max(32, int(h * self.scale))
        small = frame_bgr
        if (sw, sh) != (w, h):
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        try:
            self._net.setInputSize((sw, sh))
            _, faces = self._net.detect(small)
        except cv2.error:
            return []
        if faces is None or len(faces) == 0:
            return []
        inv = 1.0 / self.scale
        result: list[tuple[int, int, int, int]] = []
        for row in faces[: self.max_faces]:
            x, y, fw, fh = (float(v) for v in row[:4])
            if fw < self.min_size or fh < self.min_size:
                continue
            result.append(
                (
                    int(round(x * inv)),
                    int(round(y * inv)),
                    int(round((x + fw) * inv)),
                    int(round((y + fh) * inv)),
                )
            )
        return result

    def _detect_cascade(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = gray
        if self.scale != 1.0:
            small = cv2.resize(
                gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
            )
        min_side = max(8, int(self.min_size * self.scale))
        try:
            faces = self._cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_side, min_side),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error:
            return []
        if faces is None or len(faces) == 0:
            return []
        inv = 1.0 / self.scale
        return [
            (
                int(round(x * inv)),
                int(round(y * inv)),
                int(round((x + fw) * inv)),
                int(round((y + fh) * inv)),
            )
            for x, y, fw, fh in faces[: self.max_faces]
        ]


def assign_faces_to_people(
    faces: list[tuple[int, int, int, int]],
    people: list[tuple[int, tuple[float, float, float, float]]],
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Раскладывает лица по людям: лицо принадлежит рамке, где его центр."""
    assigned: dict[int, list[tuple[int, int, int, int]]] = {pid: [] for pid, _ in people}
    for face in faces:
        cx = (face[0] + face[2]) / 2.0
        cy = (face[1] + face[3]) / 2.0
        owner: int | None = None
        for pid, (x1, y1, x2, y2) in people:
            # Расширяем рамку вверх: лицо может быть чуть выше бокса.
            if x1 - 8 <= cx <= x2 + 8 and y1 - 16 <= cy <= y2:
                owner = pid
                break
        if owner is None and people:
            # Если YOLO нашёл одного человека, относим лицо к нему.
            if len(people) == 1:
                px1, py1, px2, py2 = people[0][1]
                if px1 <= cx <= px2:
                    owner = people[0][0]
        if owner is not None:
            assigned[owner].append(face)
    return assigned


def clothing_hint(
    frame_bgr: np.ndarray, box: tuple[float, float, float, float]
) -> str | None:
    return region_color(frame_bgr, box, upper=True)


def position_hint(box: tuple[float, float, float, float], frame_w: int) -> str:
    cx = (box[0] + box[2]) / 2.0
    third = frame_w / 3.0
    if cx < third:
        return "слева"
    if cx > 2 * third:
        return "справа"
    return "по центру"


def distance_hint(box: tuple[float, float, float, float], frame_h: int) -> str:
    if frame_h <= 0:
        return "?"
    ratio = max(0.0, box[3] - box[1]) / frame_h
    if ratio > 0.62:
        return "близко"
    if ratio > 0.3:
        return "средне"
    return "далеко"
