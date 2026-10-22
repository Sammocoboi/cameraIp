"""Отрисовка рамок, подписей и информационной панели (с кириллицей)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .config import class_color

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import FrameResult

log = logging.getLogger("camvision")

try:  # Pillow нужен для русского текста; без него — транслитерация.
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

_REGULAR_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
_BOLD_CANDIDATES = (
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)

_TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E",
        "Ж": "ZH", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M",
        "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
        "Ф": "F", "Х": "H", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SCH",
        "Ы": "Y", "Э": "E", "Ю": "YU", "Я": "YA", "«": '"', "»": '"',
        "—": "-", "–": "-", "·": "|", "≈": "~", "×": "x",
    }
)


def translit(text: str) -> str:
    return text.translate(_TRANSLIT)


def _first_existing(paths: tuple[str, ...]) -> str | None:
    for item in paths:
        if Path(item).exists():
            return item
    return None


@dataclass
class Text:
    xy: tuple[int, int]
    text: str
    size: int = 16
    color: tuple[int, int, int] = (245, 245, 245)  # BGR
    bg: tuple[int, int, int] | None = None
    bold: bool = False
    pad: int = 4
    anchor: str = "la"
    shadow: bool = True


class Painter:
    """Рисует русский текст: Pillow, при отсутствии — cv2 + транслит."""

    def __init__(self, regular: str | None = None, bold: str | None = None) -> None:
        self.regular = regular or _first_existing(_REGULAR_CANDIDATES)
        self.bold = bold or _first_existing(_BOLD_CANDIDATES)
        self._fonts: dict[tuple[int, bool], Any] = {}
        self._draw_probe: Any = None
        self.unicode = bool(_PIL_OK and self.regular)
        if not self.unicode:
            log.info("Pillow/шрифт не найдены — подписи будут латиницей.")

    # ------------------------------------------------------------------
    def _font(self, size: int, bold: bool) -> Any:
        size = max(8, int(size))
        key = (size, bold)
        font = self._fonts.get(key)
        if font is not None:
            return font
        path = (self.bold or self.regular) if bold else self.regular
        try:
            if path:
                font = ImageFont.truetype(path, size)
            else:
                font = ImageFont.load_default(size=size)
        except Exception:
            font = ImageFont.load_default()
        self._fonts[key] = font
        return font

    def measure(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        if not self.unicode:
            scale = size / 30.0
            (w, h), _ = cv2.getTextSize(translit(text), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            return int(w), int(h * 1.35)
        if self._draw_probe is None:
            self._draw_probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        box = self._draw_probe.textbbox((0, 0), text, font=self._font(size, bold))
        return int(box[2] - box[0]), int(box[3] - box[1])

    def truncate(self, text: str, size: int, max_width: int, bold: bool = False) -> str:
        if max_width <= 4 or self.measure(text, size, bold)[0] <= max_width:
            return text
        suffix = "…" if self.unicode else "..."
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.measure(text[:mid].rstrip() + suffix, size, bold)[0] <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + suffix

    # ------------------------------------------------------------------
    def render(self, image: np.ndarray, texts: list[Text]) -> np.ndarray:
        if not texts:
            return image
        if not self.unicode:
            return self._render_cv(image, texts)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        for item in texts:
            font = self._font(item.size, item.bold)
            fill = (item.color[2], item.color[1], item.color[0])
            if item.bg is not None:
                box = draw.textbbox(item.xy, item.text, font=font, anchor=item.anchor)
                draw.rectangle(
                    [box[0] - item.pad, box[1] - item.pad, box[2] + item.pad, box[3] + item.pad],
                    fill=(item.bg[2], item.bg[1], item.bg[0]),
                )
            if item.shadow and item.bg is None:
                draw.text(
                    (item.xy[0] + 1, item.xy[1] + 1),
                    item.text,
                    font=font,
                    fill=(0, 0, 0),
                    anchor=item.anchor,
                )
            draw.text(item.xy, item.text, font=font, fill=fill, anchor=item.anchor)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def _render_cv(self, image: np.ndarray, texts: list[Text]) -> np.ndarray:
        out = image
        for item in texts:
            scale = max(0.3, item.size / 30.0)
            text = translit(item.text)
            (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            x, y = item.xy
            if item.anchor.startswith("l"):
                pass
            else:
                x -= w // 2
            y_bottom = y + h
            if item.bg is not None:
                cv2.rectangle(
                    out,
                    (x - item.pad, y - item.pad),
                    (x + w + item.pad, y_bottom + base + item.pad),
                    item.bg,
                    -1,
                )
            cv2.putText(
                out,
                text,
                (x, y_bottom),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                item.color,
                1,
                cv2.LINE_AA,
            )
        return out


# ----------------------------------------------------------------------
def _blend_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _touch_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Чуть более светлый вариант цвета для текста."""
    b, g, r = bgr
    mix = lambda c: int(c * 0.35 + 255 * 0.65)
    return mix(b), mix(g), mix(r)


def draw_scene(
    frame: np.ndarray,
    result: "FrameResult",
    painter: Painter,
    *,
    show_panel: bool = True,
    show_fine: bool = True,
    show_faces: bool = True,
    show_hints: bool = True,
    watermark: str = "",
) -> np.ndarray:
    """Рисует результат анализа на кадре и возвращает новый кадр."""
    image = frame.copy()
    h, w = image.shape[:2]
    unit = min(w, h)
    base_font = max(12, min(22, int(unit / 43)))
    small_font = max(10, base_font - 3)

    objects = list(getattr(result, "objects", []))
    texts: list[Text] = []

    # --- лица ---------------------------------------------------------
    if show_faces:
        for face in getattr(result, "faces", []):
            x1, y1, x2, y2 = face
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 235, 120), 1, cv2.LINE_AA)

    # --- рамки объектов ----------------------------------------------
    for obj in objects:
        color = class_color(obj.name, obj.class_id)
        x1, y1, x2, y2 = obj.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # Уголки для аккуратного вида.
        length = max(10, min(26, (x2 - x1) // 5, (y2 - y1) // 5))
        for px, py, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(image, (px, py), (px + dx * length, py), color, 3, cv2.LINE_AA)
            cv2.line(image, (px, py), (px, py + dy * length), color, 3, cv2.LINE_AA)

        label = f"#{obj.track_id} {obj.ru} {obj.conf * 100:.0f}%"
        text_w, text_h = painter.measure(label, base_font, bold=True)
        lx = min(max(0, x1), max(0, w - text_w - 8))
        ly = y1 - text_h - 10 if y1 >= text_h + 14 else y1 + 2
        texts.append(
            Text((lx + 4, ly + 2), label, base_font, _touch_color(color), (16, 16, 16), True)
        )

        if show_fine and obj.hypotheses:
            top = obj.hypotheses[0]
            fine = f"≈ {top.name} {top.score * 100:.0f}%"
            if len(obj.hypotheses) > 1:
                fine += f" · {obj.hypotheses[1].name} {obj.hypotheses[1].score * 100:.0f}%"
            fine = painter.truncate(fine, small_font, max(120, min(x2 - x1 + 90, w - lx - 10)))
            texts.append(
                Text(
                    (lx + 4, ly + 2 + text_h + 8),
                    fine,
                    small_font,
                    (232, 232, 232),
                    (16, 16, 16),
                )
            )

    # --- панель со списком -------------------------------------------
    if show_panel and objects:
        panel_w = min(470, max(260, int(w * 0.37)), max(120, w - 28))
        pad = 10
        line_h = int(base_font * 1.42)
        x0 = w - panel_w - 14
        y0 = 14
        if x0 < 8:
            x0 = 8
            panel_w = max(120, w - 16)

        # Считаем высоту заранее.
        body_lines = 0
        for obj in objects[:9]:
            body_lines += 1
            body_lines += 1 if obj.hypotheses or obj.name == "person" else 0
        panel_h = min(h - 28, 58 + body_lines * line_h + 22)
        if panel_h < 60:
            show_panel = False
        else:
            _blend_rect(image, (x0, y0, x0 + panel_w, y0 + panel_h), (12, 12, 12), 0.68)
            cv2.rectangle(
                image, (x0, y0), (x0 + panel_w, y0 + panel_h), (85, 85, 85), 1, cv2.LINE_AA
            )

    if show_panel and objects and panel_h >= 60:
        cursor_y = y0 + pad
        texts.append(
            Text((x0 + pad, cursor_y), "ОБЪЕКТЫ В КАДРЕ", base_font, (255, 220, 90), bold=True)
        )
        cursor_y += int(base_font * 1.55)

        counts: dict[str, int] = {}
        for obj in objects:
            counts[obj.ru] = counts.get(obj.ru, 0) + 1
        summary = " · ".join(f"{name}: {count}" for name, count in counts.items())
        summary = painter.truncate(summary, small_font, panel_w - 2 * pad)
        texts.append(Text((x0 + pad, cursor_y), summary, small_font, (175, 175, 175)))
        cursor_y += int(line_h * 1.1)

        max_rows = max(2, int((panel_h - (cursor_y - y0) - 12) / line_h))
        for obj in objects[: max_rows - 1]:
            color = class_color(obj.name, obj.class_id)
            header = f"#{obj.track_id} {obj.ru}  {obj.conf * 100:.0f}%"
            texts.append(Text((x0 + pad, cursor_y), header, base_font, _touch_color(color), bold=True))
            cursor_y += line_h

            detail = ""
            if obj.name == "person":
                bits = []
                if obj.faces:
                    bits.append(f"лиц: {obj.faces}")
                if obj.distance:
                    bits.append(obj.distance)
                if obj.position:
                    bits.append(obj.position)
                if obj.clothing:
                    bits.append(f"одежда: {obj.clothing}")
                detail = " · ".join(bits)
            if show_fine and obj.hypotheses:
                guesses = " · ".join(
                    f"{hyp.name} {hyp.score * 100:.0f}%" for hyp in obj.hypotheses[:2]
                )
                if obj.uncertain:
                    guesses = f"не уверен · {guesses}"
                detail = guesses if not detail else f"{detail} | {guesses}"
            if detail:
                detail = painter.truncate(detail, small_font, panel_w - 2 * pad - 14)
                texts.append(Text((x0 + pad + 10, cursor_y), detail, small_font, (205, 205, 205)))
                cursor_y += line_h

        if len(objects) > max_rows - 1:
            texts.append(
                Text(
                    (x0 + pad, cursor_y),
                    f"… ещё {len(objects) - (max_rows - 1)}",
                    small_font,
                    (170, 170, 170),
                )
            )

    # --- верхняя строка и подсказки ----------------------------------
    if watermark:
        texts.append(Text((14, 12), watermark, small_font, (235, 235, 235), (0, 0, 0), pad=4))

    if show_hints:
        hint = "q выход · s снимок · f CLIP · h панель · m зеркало · пробел пауза"
        hint_w, hint_h = painter.measure(hint, small_font)
        hy = h - hint_h - 12
        _blend_rect(image, (10, hy - 5, 10 + hint_w + 14, hy + hint_h + 6), (10, 10, 10), 0.5)
        texts.append(Text((17, hy), hint, small_font, (225, 225, 225), shadow=False))

    return painter.render(image, texts)
