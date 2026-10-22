"""Работа с цветом: доминирующий оттенок, цвет одежды, эвристики."""

from __future__ import annotations

import cv2
import numpy as np

# HSV: Hue в OpenCV лежит в диапазоне 0..179.
_HUE_BINS: tuple[tuple[int, int, str], ...] = (
    (0, 10, "красный"),
    (10, 22, "оранжевый"),
    (22, 34, "жёлтый"),
    (34, 48, "жёлто-зелёный"),
    (48, 85, "зелёный"),
    (85, 100, "бирюзовый"),
    (100, 125, "голубой"),
    (125, 145, "синий"),
    (145, 160, "фиолетовый"),
    (160, 175, "розовый"),
    (175, 180, "красный"),
)


def hue_name(hue: int) -> str:
    """Название цвета по значению Hue (0..179)."""
    for lo, hi, name in _HUE_BINS:
        if lo <= hue < hi:
            return name
    return "неопределённый"


def _name_from_hsv(h: int, s: int, v: int) -> str:
    name = hue_name(h)
    if s < 45:
        return "белый" if v > 190 else ("серый" if v > 90 else "чёрный")
    if v < 65:
        return f"тёмно-{name}"
    if v > 215 and s < 90:
        return f"светло-{name}"
    return name


def region_color(
    frame_bgr: np.ndarray,
    xyxy: tuple[float, float, float, float],
    *,
    upper: bool = False,
    min_saturation: int = 45,
    min_value: int = 35,
) -> str | None:
    """Основной цвет области (например, одежды человека).

    ``upper=True`` берёт верхнюю половину рамки — так цвет одежды
    определяется стабильнее, чем по всему силуэту.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    if upper:
        y2 = y1 + max(4, int((y2 - y1) * 0.45))

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)
    if int(mask.sum()) < 0.12 * mask.size:
        # Почти нет насыщенных пикселей — значит, белое/серое/чёрное.
        mask = np.ones(hsv.shape[:2], dtype=bool)

    h_med = int(np.median(hsv[:, :, 0][mask]))
    s_med = int(np.median(hsv[:, :, 1][mask]))
    v_med = int(np.median(hsv[:, :, 2][mask]))
    return _name_from_hsv(h_med, s_med, v_med)


def dominant_color(
    frame_bgr: np.ndarray,
    *,
    min_saturation: int = 55,
    min_value: int = 40,
) -> tuple[str | None, float]:
    """Доминирующий насыщенный цвет и его доля.

    Возвращает ``(None, 0.0)``, если картинка почти монохромная.
    Доля считается по насыщенным пикселям области.
    """
    if frame_bgr.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)
    count = int(mask.sum())
    if count < 0.06 * mask.size:
        return None, 0.0

    hues = hsv[:, :, 0][mask]
    hist = np.bincount(hues, minlength=180).astype(np.float32) / count
    # Сглаживаем гистограмму, чтобы не «прыгать» между соседними оттенками.
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    hist = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), kernel, mode="same")[2:-2]
    hue = int(np.argmax(hist))
    share = float(hist[hue])
    if share < 0.16:
        return None, 0.0
    return hue_name(hue), share


def hex_bgr(color: tuple[int, int, int]) -> str:
    b, g, r = color
    return f"#{r:02x}{g:02x}{b:02x}"
