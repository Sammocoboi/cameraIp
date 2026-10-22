"""Сборка кадра: детекция → трекинг → вид/модель → лица → результат."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .faces import (
    FaceDetector,
    assign_faces_to_people,
    clothing_hint,
    distance_hint,
    position_hint,
)
from .finegrained import Hypothesis, LabelBook, PlantHeuristics
from .config import class_color  # noqa: F401  (используется в отрисовке)
from .tracker import IouTracker, Track

log = logging.getLogger("camvision")


@dataclass
class SceneObject:
    track_id: int
    name: str
    ru: str
    class_id: int
    conf: float
    xyxy: tuple[int, int, int, int]
    hypotheses: list[Hypothesis] = field(default_factory=list)
    uncertain: bool = False
    faces: int = 0
    clothing: str | None = None
    position: str | None = None
    distance: str | None = None
    age: int = 0


@dataclass
class FrameResult:
    objects: list[SceneObject] = field(default_factory=list)
    faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    fps: float = 0.0
    infer_ms: float = 0.0
    fine_ms: float = 0.0
    person_count: int = 0
    events: list[tuple[str, SceneObject]] = field(default_factory=list)
    detector_name: str = ""
    classifier_name: str = ""
    frame_index: int = 0


@dataclass
class AnalyzerOptions:
    fg_enabled: bool = True
    fg_topk: int = 3
    fg_samples: int = 4
    fg_refresh_s: float = 6.0
    fg_min_crop: int = 40
    fg_sample_gap_s: float = 0.15
    fg_margin: float | None = None
    faces_enabled: bool = True
    face_every: int = 3


def crop_box(
    frame: np.ndarray, box: tuple[float, float, float, float], expand: float = 0.08
) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= bw * expand
    y1 -= bh * expand
    x2 += bw * expand
    y2 += bh * expand
    x1i = max(0, min(w - 1, int(round(x1))))
    y1i = max(0, min(h - 1, int(round(y1))))
    x2i = max(x1i + 1, min(w, int(round(x2))))
    y2i = max(y1i + 1, min(h, int(round(y2))))
    return frame[y1i:y2i, x1i:x2i]


class Analyzer:
    """Оркестратор всего конвейера. Детектор и классификатор внедряются."""

    def __init__(
        self,
        detector: Any,
        labels: LabelBook,
        *,
        classifier: Any | None = None,
        face_detector: FaceDetector | None = None,
        tracker: IouTracker | None = None,
        options: AnalyzerOptions | None = None,
    ) -> None:
        self.detector = detector
        self.labels = labels
        self.classifier = classifier
        self.faces = face_detector
        self.tracker = tracker or IouTracker()
        self.options = options or AnalyzerOptions()
        self.heuristics = PlantHeuristics()

        self._frame_idx = 0
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self._announced: set[int] = set()
        self._fg_announced: dict[int, str] = {}
        self._last_embed_at = 0.0
        self._face_cache: list[tuple[tuple[int, int, int, int], int | None, tuple[float, float, float, float]]] = []
        self._last_result = FrameResult()

    # ------------------------------------------------------------------
    @property
    def fg_enabled(self) -> bool:
        return self.options.fg_enabled

    @fg_enabled.setter
    def fg_enabled(self, value: bool) -> None:
        self.options.fg_enabled = bool(value)

    def reset(self) -> None:
        self.tracker.reset()
        self._announced.clear()
        self._fg_announced.clear()
        self._face_cache.clear()
        self._last_result = FrameResult()

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, *, detect: bool = True) -> FrameResult:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0 else self._fps * 0.85 + instant * 0.15

        frame_h, frame_w = frame_bgr.shape[:2]
        infer_ms = 0.0
        fine_ms = 0.0

        if detect:
            self._frame_idx += 1
            t0 = time.perf_counter()
            detections = self.detector.detect(frame_bgr)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            tracks = self.tracker.update(detections, self._frame_idx)
        else:
            tracks = list(self.tracker.tracks)

        person_tracks = [t for t in tracks if t.name == "person"]
        if detect and self.options.faces_enabled and self.faces is not None and self.faces.ok:
            if self._frame_idx % self.options.face_every == 0:
                t0 = time.perf_counter()
                found = self.faces.detect(frame_bgr)
                infer_ms += (time.perf_counter() - t0) * 1000.0
                people = [(t.id, t.xyxy) for t in person_tracks]
                assigned = assign_faces_to_people(found, people)
                anchor = {t.id: t.xyxy for t in person_tracks}
                cache: list = []
                for pid, faces in assigned.items():
                    track = self._track_by_id(tracks, pid)
                    if track is not None:
                        track.faces = len(faces)
                        track.faces_at_frame = self._frame_idx
                    for face in faces:
                        cache.append((face, pid, anchor.get(pid)))
                # Лица, не попавшие ни в одного человека, тоже показываем.
                assigned_faces = {f for faces in assigned.values() for f in faces}
                for face in found:
                    if face not in assigned_faces:
                        cache.append((face, None, None))
                self._face_cache = cache

        visible_faces = self._shift_faces(tracks)

        if detect:
            fine_ms = self._update_fine_grained(frame_bgr, tracks, now)

        objects: list[SceneObject] = []
        for track in sorted(tracks, key=lambda t: (t.name != "person", t.id)):
            if track.missed > 3:
                continue
            if track.name == "person":
                if track.clothing is None and track.hits >= 2:
                    track.clothing = clothing_hint(frame_bgr, track.xyxy)
                obj_faces = track.faces if (self._frame_idx - track.faces_at_frame) <= 8 else 0
            else:
                obj_faces = 0
            objects.append(
                SceneObject(
                    track_id=track.id,
                    name=track.name,
                    ru=track.ru,
                    class_id=track.cls,
                    conf=track.conf,
                    xyxy=track.box,
                    hypotheses=list(track.hypotheses),
                    uncertain=track.uncertain,
                    faces=obj_faces,
                    clothing=track.clothing,
                    position=position_hint(track.xyxy, frame_w),
                    distance=distance_hint(track.xyxy, frame_h),
                    age=track.age,
                )
            )

        events: list[tuple[str, SceneObject]] = []
        for obj in objects:
            if obj.track_id not in self._announced:
                self._announced.add(obj.track_id)
                events.append(("new", obj))
            if obj.hypotheses:
                key = obj.hypotheses[0].name
                if self._fg_announced.get(obj.track_id) != key:
                    self._fg_announced[obj.track_id] = key
                    events.append(("fine", obj))

        result = FrameResult(
            objects=objects,
            faces=visible_faces,
            fps=self._fps,
            infer_ms=infer_ms,
            fine_ms=fine_ms,
            person_count=sum(1 for o in objects if o.name == "person"),
            events=events,
            detector_name=str(getattr(self.detector, "describe", lambda: "detector")()),
            classifier_name=(
                str(getattr(self.classifier, "short_name", "")) if self.classifier is not None else ""
            ),
            frame_index=self._frame_idx,
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _track_by_id(tracks: list[Track], track_id: int) -> Track | None:
        for track in tracks:
            if track.id == track_id:
                return track
        return None

    def _shift_faces(self, tracks: list[Track]) -> list[tuple[int, int, int, int]]:
        by_id = {t.id: t for t in tracks}
        out: list[tuple[int, int, int, int]] = []
        for face, pid, anchor in self._face_cache:
            track = by_id.get(pid) if pid is not None else None
            if pid is not None and track is None:
                continue
            dx = dy = 0.0
            if track is not None and anchor is not None:
                dx = (track.xyxy[0] + track.xyxy[2]) / 2 - (anchor[0] + anchor[2]) / 2
                dy = (track.xyxy[1] + track.xyxy[3]) / 2 - (anchor[1] + anchor[3]) / 2
            out.append(
                (
                    int(round(face[0] + dx)),
                    int(round(face[1] + dy)),
                    int(round(face[2] + dx)),
                    int(round(face[3] + dy)),
                )
            )
        return out

    # ------------------------------------------------------------------
    def _update_fine_grained(
        self, frame: np.ndarray, tracks: list[Track], now: float
    ) -> float:
        if not self.options.fg_enabled:
            return 0.0
        group = None
        spent_ms = 0.0
        for track in tracks:
            if track.missed > 0:
                continue
            group = self.labels.group_for(track.name)
            if group is None:
                continue
            if track.width < self.options.fg_min_crop or track.height < self.options.fg_min_crop:
                continue

            # 1. Набираем несколько эмбеддингов, усредняем.
            if (
                track.samples < self.options.fg_samples
                and now - self._last_embed_at >= self.options.fg_sample_gap_s
                and self.classifier is not None
                and getattr(self.classifier, "available", False)
            ):
                crop = crop_box(frame, track.xyxy)
                t0 = time.perf_counter()
                embedding = self.classifier.embed(crop)
                spent_ms += (time.perf_counter() - t0) * 1000.0
                self._last_embed_at = now
                if embedding is not None:
                    track.embedding = self._merge_embedding(track.embedding, embedding, track.samples)
                    track.samples += 1
                    track.last_sample_frame = self._frame_idx

            classifier_ready = self.classifier is not None and getattr(
                self.classifier, "available", False
            )
            enough_samples = track.samples >= min(2, self.options.fg_samples)
            # 2. Классифицируем (не чаще, чем раз в fg_refresh_s).
            if (enough_samples or (group.kind == "plant" and not classifier_ready)) and (
                not track.fg_ready or now - track.fg_checked_at >= self.options.fg_refresh_s
            ):
                if classifier_ready and track.embedding is None:
                    continue
                t0 = time.perf_counter()
                if classifier_ready:
                    result = self.classifier.classify(
                        group,
                        track.embedding,
                        topk=self.options.fg_topk,
                        margin=self.options.fg_margin,
                    )
                    track.hypotheses = result.hypotheses
                    track.uncertain = result.uncertain
                elif group.kind == "plant":
                    track.hypotheses = self.heuristics.guess_plant(
                        crop_box(frame, track.xyxy)
                    )
                    track.uncertain = True
                track.fg_checked_at = now
                track.fg_ready = bool(track.hypotheses)
                spent_ms += (time.perf_counter() - t0) * 1000.0
        return spent_ms

    @staticmethod
    def _merge_embedding(
        old: np.ndarray | None, new: np.ndarray, count: int
    ) -> np.ndarray:
        if old is None:
            merged = new.astype(np.float32)
        else:
            merged = (old.astype(np.float32) * count + new.astype(np.float32)) / (count + 1)
        norm = float(np.linalg.norm(merged)) + 1e-9
        return (merged / norm).astype(np.float32)
