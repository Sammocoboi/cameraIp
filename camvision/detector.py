"""Обнаружение объектов через Ultralytics YOLO."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import MODELS_DIR, class_ru

log = logging.getLogger("camvision")


class DetectorError(RuntimeError):
    """Не удалось загрузить модель детектора."""


@dataclass(slots=True)
class Detection:
    """Одно найденное окно."""

    xyxy: tuple[int, int, int, int]
    conf: float
    class_id: int
    name: str
    ru: str
    track_id: int | None = None


def _resolve_model_path(model: str) -> str:
    """Кладём скачанные веса в models/, а не в корень проекта."""
    path = Path(model)
    if path.exists():
        return str(path)
    if "://" in model or path.is_absolute():
        return model
    if len(path.parts) > 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / path.name)


class YoloDetector:
    """Тонкая обёртка над ``ultralytics.YOLO``."""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        *,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        device: str | None = None,
        classes: Sequence[int] | None = None,
        max_det: int = 40,
    ) -> None:
        self.model_spec = model
        self.model_path = _resolve_model_path(model)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.classes = list(classes) if classes else None
        self.max_det = int(max_det)
        self.model = None
        self.names: dict[int, str] = {}

    # ------------------------------------------------------------------
    def load(self) -> "YoloDetector":
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise DetectorError(
                "Не установлен ultralytics. Выполните:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        log.info("Загружаю детектор %s ...", self.model_path)
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            hint = (
                f"Не удалось загрузить модель '{self.model_spec}'.\n"
                "Проверьте интернет (веса скачиваются при первом запуске) или укажите\n"
                "локальный файл: --model path/to/yolo11n.pt\n"
                "Список моделей: https://docs.ultralytics.com/models/"
            )
            raise DetectorError(hint) from exc

        names = getattr(self.model, "names", {}) or {}
        self.names = {int(k): str(v) for k, v in dict(names).items()}
        if not self.names:
            self.names = {i: f"class_{i}" for i in range(80)}
        return self

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()
        h, w = frame_bgr.shape[:2]
        kwargs: dict = {
            "source": frame_bgr,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.classes:
            kwargs["classes"] = self.classes

        results = self.model.predict(**kwargs)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(np.float32)
        classes = boxes.cls.cpu().numpy().astype(np.int32)

        out: list[Detection] = []
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, classes):
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(0, min(w - 1, int(round(x2))))
            y2 = max(0, min(h - 1, int(round(y2))))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            name = self.names.get(int(cid), f"class_{int(cid)}")
            out.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    conf=float(conf),
                    class_id=int(cid),
                    name=name,
                    ru=class_ru(name),
                )
            )
        return out

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return f"YOLO ({Path(self.model_path).name})"


class FakeDetector:
    """Детектор-заглушка для тестов и отладки интерфейса."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:  # noqa: ARG002
        return list(self._detections)

    def describe(self) -> str:
        return "fake"
