"""Сохранение результатов: JSONL, видео, снимки."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pipeline import FrameResult

log = logging.getLogger("camvision")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class JsonlWriter:
    """Потоковый JSONL: одна строка на кадр/событие."""

    def __init__(self, path: str | Path, *, every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, result: FrameResult, *, source: str = "", annotate: bool = True) -> None:
        if result.frame_index % self.every != 0:
            return
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "frame": result.frame_index,
            "source": source,
            "fps": round(result.fps, 2),
            "infer_ms": round(result.infer_ms + result.fine_ms, 1),
            "people": result.person_count,
            "objects": [],
        }
        if annotate:
            for obj in result.objects:
                payload["objects"].append(
                    {
                        "id": obj.track_id,
                        "class": obj.name,
                        "ru": obj.ru,
                        "confidence": round(obj.conf, 4),
                        "bbox": list(obj.xyxy),
                        "faces": obj.faces,
                        "clothing": obj.clothing,
                        "distance": obj.distance,
                        "position": obj.position,
                        "model": [
                            {"name": h.name, "score": round(h.score, 4), "source": h.source}
                            for h in obj.hypotheses
                        ],
                        "model_uncertain": obj.uncertain,
                    }
                )
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class VideoRecorder:
    """Запись аннотированного видео в mp4."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, max(1.0, float(fps)), size)
        self.ok = self.writer.isOpened()
        if self.ok:
            log.info("Запись видео: %s", self.path)
        else:
            log.warning("Не удалось открыть видеофайл для записи: %s", self.path)

    def write(self, frame: np.ndarray) -> None:
        if self.ok:
            self.writer.write(frame)

    def close(self) -> None:
        if self.ok:
            self.writer.release()
            self.ok = False
            log.info("Видео сохранено: %s", self.path)


def save_snapshot(
    frame: np.ndarray, directory: str | Path, prefix: str = "snapshot"
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{timestamp()}.jpg"
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Не удалось закодировать снимок")
    path.write_bytes(encoded.tobytes())
    return path
