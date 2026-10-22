"""Простой IoU-трекер: стабильные ID объектов между кадрами."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection
from .finegrained import Hypothesis


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """Объект, который живёт несколько кадров."""

    id: int
    cls: int
    name: str
    ru: str
    xyxy: tuple[float, float, float, float]
    conf: float
    age: int = 1
    hits: int = 1
    missed: int = 0
    first_frame: int = 0
    last_frame: int = 0
    # Кэш «мелкой» классификации (CLIP).
    embedding: object | None = None
    samples: int = 0
    last_sample_frame: int = -999
    hypotheses: list[Hypothesis] = field(default_factory=list)
    uncertain: bool = False
    fg_checked_at: float = 0.0
    fg_ready: bool = False
    # Люди.
    faces: int = 0
    clothing: str | None = None
    faces_at_frame: int = -999

    @property
    def box(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])


class IouTracker:
    """Жадное сопоставление по IoU внутри одного класса.

    Этого достаточно для задачи «кто есть кто» на видео с камеры:
    без Калмана, но с сохранением метки при коротких пропаданиях.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 25, smooth: float = 0.65) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.smooth = float(smooth)
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det.class_id != track.cls:
                    continue
                value = iou(track.xyxy, det.xyxy)
                if value >= self.iou_threshold:
                    pairs.append((value, ti, di))
        pairs.sort(key=lambda item: item[0], reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            track, det = self.tracks[ti], detections[di]
            a = self.smooth
            track.xyxy = (
                a * det.xyxy[0] + (1 - a) * track.xyxy[0],
                a * det.xyxy[1] + (1 - a) * track.xyxy[1],
                a * det.xyxy[2] + (1 - a) * track.xyxy[2],
                a * det.xyxy[3] + (1 - a) * track.xyxy[3],
            )
            track.conf = det.conf
            track.missed = 0
            track.hits += 1
            track.age += 1
            track.last_frame = frame_idx
            det.track_id = track.id
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = Track(
                id=self._next_id,
                cls=det.class_id,
                name=det.name,
                ru=det.ru,
                xyxy=tuple(float(v) for v in det.xyxy),  # type: ignore[arg-type]
                conf=det.conf,
                first_frame=frame_idx,
                last_frame=frame_idx,
            )
            det.track_id = track.id
            self.tracks.append(track)
            self._next_id += 1

        alive: list[Track] = []
        for track in self.tracks:
            if track.last_frame != frame_idx:
                track.missed += 1
                track.age += 1
                if track.missed > self.max_age:
                    continue
            alive.append(track)
        self.tracks = alive
        return list(self.tracks)
