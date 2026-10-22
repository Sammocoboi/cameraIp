"""Определение видов и моделей: CLIP zero-shot + цветовые эвристики.

Идея: детектор говорит «это ноутбук», а CLIP сравнивает вырезанный объект
с описаниями конкретных моделей (MacBook Pro, ThinkPad, …) и видов растений.
Это именно *предположения*: CLIP не читает логотипы, поэтому результат
показывается как top-N с вероятностями и явной пометкой «не уверен».
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .colors import dominant_color
from .config import CACHE_DIR, LABELS_PATH

log = logging.getLogger("camvision")


@dataclass(slots=True)
class Hypothesis:
    """Одна гипотеза «это может быть …»."""

    name: str
    score: float
    source: str = "clip"  # clip | эвристика


@dataclass(slots=True)
class FineResult:
    hypotheses: list[Hypothesis]
    generic_similarity: float | None = None
    uncertain: bool = False


@dataclass
class LabelSpec:
    ru: str
    prompts: list[str]


@dataclass
class LabelGroup:
    key: str
    title: str
    targets: tuple[str, ...]
    kind: str
    labels: list[LabelSpec]
    generic: list[str] = field(default_factory=list)
    margin: float = 0.012

    def find(self, name: str) -> float:
        for label in self.labels:
            if label.ru == name:
                return 1.0
        return 0.0


class LabelBook:
    """Списки моделей/видов из labels/finegrained.json."""

    def __init__(self, path: str | Path = LABELS_PATH) -> None:
        self.path = Path(path)
        self.groups: list[LabelGroup] = []
        self._by_target: dict[str, LabelGroup] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            log.warning("Файл подсказок не найден: %s", self.path)
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for raw in data.get("groups", []):
            kind = str(raw.get("kind", "product"))
            templates = list(raw.get("templates") or DEFAULT_TEMPLATES[kind])
            labels: list[LabelSpec] = []
            for item in raw.get("labels", []):
                ru = str(item.get("ru") or item.get("en") or "?")
                en = str(item.get("en") or ru)
                labels.append(LabelSpec(ru=ru, prompts=[t.format(label=en) for t in templates]))
            if not labels:
                continue
            generic = str(raw.get("generic") or "")
            group = LabelGroup(
                key=str(raw.get("key") or raw.get("title") or "group"),
                title=str(raw.get("title") or "Варианты"),
                targets=tuple(str(t) for t in raw.get("targets", [])),
                kind=kind,
                labels=labels,
                generic=[t.format(label=generic) for t in templates] if generic else [],
                margin=float(raw.get("margin", 0.012)),
            )
            self.groups.append(group)
            for target in group.targets:
                self._by_target[target] = group

    def group_for(self, coco_name: str) -> LabelGroup | None:
        return self._by_target.get(coco_name)

    def __len__(self) -> int:
        return len(self.groups)


DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "product": [
        "a photo of {label}",
        "a close-up product photo of {label}",
        "{label} on a desk, product shot",
    ],
    "plant": [
        "a photo of {label}",
        "a close-up photo of {label}",
        "{label} growing as a houseplant",
    ],
}


class ClipClassifier:
    """Zero-shot классификатор на open_clip."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        *,
        device: str | None = None,
        cache_dir: str | Path = CACHE_DIR,
        temperature: float = 0.02,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device_pref = device or "auto"
        self.cache_dir = Path(cache_dir)
        self.temperature = float(temperature)
        self.log = logger or log
        self.device = "cpu"
        self.error: str | None = None
        self.backend_name = f"CLIP {model_name}"
        self._attempted = False
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._text_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.ensure_loaded()

    def ensure_loaded(self) -> bool:
        if self._attempted:
            return self.error is None
        self._attempted = True
        try:
            import torch
            import open_clip
        except Exception as exc:
            self.error = (
                "CLIP не установлен (pip install open_clip_torch). "
                "Определение видов/моделей отключено."
            )
            self.log.warning("%s (%s)", self.error, exc)
            return False

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            if self.device_pref == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.device_pref
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            self.log.info("Загружаю CLIP %s / %s ...", self.model_name, self.pretrained)
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device,
                    cache_dir=str(self.cache_dir),
                )
            except TypeError:  # старые версии open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device,
                )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._torch = torch
        except Exception as exc:
            self.error = (
                "Не удалось загрузить веса CLIP. Проверьте интернет "
                f"или кэш {self.cache_dir}. Ошибка: {exc}"
            )
            self.log.warning(self.error)
            self._model = None
            return False
        return True

    # ------------------------------------------------------------------
    @property
    def short_name(self) -> str:
        return f"CLIP {self.model_name}"

    def _group_features(self, group: LabelGroup) -> tuple[np.ndarray, np.ndarray | None]:
        cached = self._text_cache.get(group.key)
        if cached is not None:
            return cached

        prompts: list[str] = []
        owners: list[int] = []
        for idx, label in enumerate(group.labels):
            for prompt in label.prompts:
                prompts.append(prompt)
                owners.append(idx)
        generic_start = len(prompts)
        for prompt in group.generic:
            prompts.append(prompt)
            owners.append(-1)

        torch = self._torch
        assert torch is not None
        with torch.inference_mode():
            tokens = self._tokenizer(prompts).to(self.device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        feats = features.float().cpu().numpy()
        owners_arr = np.asarray(owners, dtype=np.int32)

        matrix = np.zeros((len(group.labels), feats.shape[1]), dtype=np.float32)
        for idx in range(len(group.labels)):
            vectors = feats[owners_arr == idx]
            mean = vectors.mean(axis=0)
            matrix[idx] = mean / (np.linalg.norm(mean) + 1e-9)

        generic_vec: np.ndarray | None = None
        if len(prompts) > generic_start:
            vectors = feats[owners_arr == -1]
            mean = vectors.mean(axis=0)
            generic_vec = (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)

        self._text_cache[group.key] = (matrix, generic_vec)
        return matrix, generic_vec

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Нормализованный эмбеддинг вырезанного объекта (BGR)."""
        if not self.ensure_loaded() or crop_bgr.size == 0:
            return None
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        torch = self._torch
        assert torch is not None
        try:
            with torch.inference_mode():
                tensor = self._preprocess(image).unsqueeze(0).to(self.device)
                feat = self._model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.float().cpu().numpy()[0]
        except Exception as exc:  # pragma: no cover - редкие ошибки CUDA
            self.log.warning("CLIP не смог обработать фрагмент: %s", exc)
            return None

    def classify(
        self,
        group: LabelGroup,
        embedding: np.ndarray,
        *,
        topk: int = 3,
        margin: float | None = None,
    ) -> FineResult:
        if not self.ensure_loaded():
            return FineResult([])
        matrix, generic_vec = self._group_features(group)
        sims = matrix @ embedding.astype(np.float32)

        temperature = max(1e-4, self.temperature)
        z = (sims - float(sims.max())) / temperature
        probs = np.exp(z)
        probs /= probs.sum()

        order = np.argsort(-sims)[: max(1, int(topk))]
        hypotheses = [
            Hypothesis(group.labels[int(i)].ru, float(probs[int(i)]), "clip") for i in order
        ]

        generic_sim = float(generic_vec @ embedding) if generic_vec is not None else None
        limit = group.margin if margin is None else float(margin)
        uncertain = generic_sim is not None and (float(sims[order[0]]) - generic_sim) < limit
        return FineResult(hypotheses, generic_sim, uncertain)

    def classify_all(
        self, groups: list[LabelGroup], embedding: np.ndarray, *, topk: int = 3
    ) -> dict[str, FineResult]:
        return {g.key: self.classify(g, embedding, topk=topk) for g in groups}


class PlantHeuristics:
    """Запасной вариант без CLIP: хоть какой-то намёк по цвету."""

    def guess_plant(self, crop_bgr: np.ndarray) -> list[Hypothesis]:
        color, share = dominant_color(crop_bgr)
        if color is None:
            return []
        confidence = min(0.35, 0.05 + share * 0.45)
        return [
            Hypothesis(
                name=f"возможен цветок ({color})",
                score=float(confidence),
                source="эвристика",
            )
        ]
