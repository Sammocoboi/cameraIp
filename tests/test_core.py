"""Быстрые тесты без камеры и без скачивания весов."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from camvision.detector import Detection
from camvision.drawing import Painter, Text, draw_scene
from camvision.faces import assign_faces_to_people
from camvision.finegrained import (
    FineResult,
    Hypothesis,
    LabelBook,
    PlantHeuristics,
)
from camvision.output import JsonlWriter
from camvision.pipeline import Analyzer, AnalyzerOptions
from camvision.tracker import IouTracker, iou


def make_detection(
    box: tuple[int, int, int, int],
    cls: int,
    name: str,
    ru: str,
    conf: float = 0.9,
) -> Detection:
    return Detection(xyxy=box, conf=conf, class_id=cls, name=name, ru=ru)


class FakeDetector:
    describe = lambda self: "fake-yolo"  # noqa: E731

    def detect(self, frame: np.ndarray) -> list[Detection]:  # noqa: ARG002
        return [
            make_detection((10, 20, 110, 220), 0, "person", "человек", 0.93),
            make_detection((200, 120, 340, 220), 63, "laptop", "ноутбук", 0.86),
        ]


class FakeClassifier:
    available = True
    short_name = "fake-clip"

    def embed(self, crop: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        vector = np.ones(8, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def classify(self, group, embedding, *, topk=3, margin=None) -> FineResult:  # noqa: ANN001, ARG002
        name = "MacBook Pro" if group.key == "laptops" else "Монстера"
        return FineResult([Hypothesis(name, 0.61), Hypothesis("другое", 0.22)], 0.1, False)


class TestTracker(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)
        self.assertGreater(iou((0, 0, 10, 10), (5, 0, 15, 10)), 0.3)

    def test_stable_id_and_expiry(self) -> None:
        tracker = IouTracker(max_age=3)
        det = make_detection((10, 10, 60, 60), 0, "person", "человек")
        tracks = tracker.update([det], 1)
        self.assertEqual(len(tracks), 1)
        first_id = tracks[0].id

        tracks = tracker.update(
            [make_detection((12, 11, 62, 61), 0, "person", "человек")], 2
        )
        self.assertEqual(tracks[0].id, first_id)
        self.assertEqual(tracks[0].hits, 2)

        for frame in range(3, 6):
            tracker.update([], frame)
        self.assertEqual([t.id for t in tracker.tracks], [first_id])

        tracker.update([], 20)
        self.assertEqual(tracker.tracks, [])

    def test_different_classes_do_not_merge(self) -> None:
        tracker = IouTracker()
        tracker.update([make_detection((10, 10, 60, 60), 0, "person", "человек")], 1)
        tracks = tracker.update([make_detection((10, 10, 60, 60), 63, "laptop", "ноутбук")], 2)
        # Старый трек человека ещё жив (max_age), но новый объект получил свой ID.
        self.assertEqual(len(tracks), 2)
        self.assertEqual({t.cls for t in tracks}, {0, 63})


class TestLabels(unittest.TestCase):
    def test_builtin_catalog(self) -> None:
        book = LabelBook()
        self.assertGreaterEqual(len(book), 3)
        for key in ("laptops", "mice", "plants"):
            group = next((g for g in book.groups if g.key == key), None)
            self.assertIsNotNone(group, key)
            self.assertGreater(len(group.labels), 5)
            self.assertTrue(group.generic, "нужен generic-промпт для проверки уверенности")
            self.assertTrue(group.labels[0].prompts[0].startswith("a photo of"))

    def test_target_mapping(self) -> None:
        book = LabelBook()
        plants = book.group_for("potted plant")
        self.assertIsNotNone(plants)
        self.assertEqual(plants.key, "plants")
        self.assertIs(book.group_for("vase"), plants)
        self.assertIsNone(book.group_for("bicycle"))

    def test_broken_file_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text(json.dumps({"groups": []}), encoding="utf-8")
            self.assertEqual(len(LabelBook(path)), 0)


class TestHeuristics(unittest.TestCase):
    def test_red_flower_hint(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:, :] = (0, 0, 220)
        guesses = PlantHeuristics().guess_plant(image)
        self.assertTrue(guesses)
        self.assertIn("красн", guesses[0].name)
        self.assertEqual(guesses[0].source, "эвристика")

    def test_gray_image_has_no_hint(self) -> None:
        image = np.full((40, 40, 3), 128, dtype=np.uint8)
        self.assertEqual(PlantHeuristics().guess_plant(image), [])


class TestPainter(unittest.TestCase):
    def test_cyrillic_text_renders(self) -> None:
        painter = Painter()
        image = np.zeros((240, 480, 3), dtype=np.uint8)
        result = painter.render(image, [Text((20, 20), "Привет, камера!", 20, (255, 255, 255))])
        self.assertEqual(result.shape, image.shape)
        self.assertGreater(int(result.sum()), 0)

    def test_truncate(self) -> None:
        painter = Painter()
        short = painter.truncate("очень длинная подпись объекта", 14, 60)
        self.assertLessEqual(painter.measure(short, 14)[0], 60)
        self.assertIn("…" if painter.unicode else ".", short)


class TestPipeline(unittest.TestCase):
    def _analyzer(self) -> Analyzer:
        options = AnalyzerOptions(
            fg_enabled=True,
            fg_samples=2,
            fg_sample_gap_s=0.0,
            faces_enabled=False,
        )
        return Analyzer(FakeDetector(), LabelBook(), classifier=FakeClassifier(), options=options)

    def test_full_frame(self) -> None:
        analyzer = self._analyzer()
        frame = np.full((360, 480, 3), 40, dtype=np.uint8)
        result = None
        kinds: list[str] = []
        for _ in range(4):
            result = analyzer.process(frame)
            kinds.extend(kind for kind, _ in result.events)
        assert result is not None

        self.assertEqual(len(result.objects), 2)
        self.assertEqual(result.person_count, 1)
        self.assertEqual([o.name for o in result.objects], ["person", "laptop"])
        self.assertEqual(len({o.track_id for o in result.objects}), 2)

        laptop = next(o for o in result.objects if o.name == "laptop")
        self.assertTrue(laptop.hypotheses)
        self.assertAlmostEqual(laptop.hypotheses[0].score, 0.61)

        self.assertIn("new", kinds)
        self.assertIn("fine", kinds)

        annotated = draw_scene(frame, result, Painter())
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.any(annotated != frame))

    def test_skip_detection_keeps_objects(self) -> None:
        analyzer = self._analyzer()
        frame = np.full((360, 480, 3), 40, dtype=np.uint8)
        analyzer.process(frame)
        result = analyzer.process(frame, detect=False)
        self.assertEqual(len(result.objects), 2)


class TestFaces(unittest.TestCase):
    def test_assignment(self) -> None:
        people = [(1, (100, 50, 300, 400)), (2, (500, 60, 700, 420))]
        faces = [(180, 90, 220, 130), (560, 100, 600, 140), (10, 10, 30, 30)]
        assigned = assign_faces_to_people(faces, people)
        self.assertEqual(len(assigned[1]), 1)
        self.assertEqual(len(assigned[2]), 1)
        self.assertEqual(sum(len(v) for v in assigned.values()), 2)


class TestJsonl(unittest.TestCase):
    def test_writes_json(self) -> None:
        analyzer = TestPipeline()._analyzer()
        frame = np.full((240, 320, 3), 30, dtype=np.uint8)
        analyzer.process(frame)
        result = analyzer.process(frame)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            writer = JsonlWriter(path)
            writer.write(result, source="test")
            writer.close()
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertIn("objects", payload)
        self.assertIn("человек", {obj["ru"] for obj in payload["objects"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
