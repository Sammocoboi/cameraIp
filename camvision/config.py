"""Общие настройки: русские названия классов COCO, цвета, пути."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "labels" / "finegrained.json"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"

# Классы COCO (индексы совпадают с YOLO, обученной на COCO).
COCO_RU: dict[str, str] = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "автомобиль",
    "motorcycle": "мотоцикл",
    "airplane": "самолёт",
    "bus": "автобус",
    "train": "поезд",
    "truck": "грузовик",
    "boat": "лодка",
    "traffic light": "светофор",
    "fire hydrant": "пожарный гидрант",
    "stop sign": "знак «стоп»",
    "parking meter": "паркомат",
    "bench": "скамейка",
    "bird": "птица",
    "cat": "кошка",
    "dog": "собака",
    "horse": "лошадь",
    "sheep": "овца",
    "cow": "корова",
    "elephant": "слон",
    "bear": "медведь",
    "zebra": "зебра",
    "giraffe": "жираф",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "handbag": "сумка",
    "tie": "галстук",
    "suitcase": "чемодан",
    "frisbee": "фрисби",
    "skis": "лыжи",
    "snowboard": "сноуборд",
    "sports ball": "спортивный мяч",
    "kite": "воздушный змей",
    "baseball bat": "бейсбольная бита",
    "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд",
    "surfboard": "доска для сёрфинга",
    "tennis racket": "теннисная ракетка",
    "bottle": "бутылка",
    "wine glass": "бокал",
    "cup": "чашка",
    "fork": "вилка",
    "knife": "нож",
    "spoon": "ложка",
    "bowl": "миска",
    "banana": "банан",
    "apple": "яблоко",
    "sandwich": "бутерброд",
    "orange": "апельсин",
    "broccoli": "брокколи",
    "carrot": "морковь",
    "hot dog": "хот-дог",
    "pizza": "пицца",
    "donut": "пончик",
    "cake": "торт",
    "chair": "стул",
    "couch": "диван",
    "potted plant": "комнатное растение",
    "bed": "кровать",
    "dining table": "обеденный стол",
    "toilet": "унитаз",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "смартфон",
    "microwave": "микроволновка",
    "oven": "духовка",
    "toaster": "тостер",
    "sink": "раковина",
    "refrigerator": "холодильник",
    "book": "книга",
    "clock": "часы",
    "vase": "ваза",
    "scissors": "ножницы",
    "teddy bear": "плюшевый мишка",
    "hair drier": "фен",
    "toothbrush": "зубная щётка",
}


def class_ru(name: str) -> str:
    """Русское имя класса с запасным вариантом."""
    return COCO_RU.get(name, name)


# BGR-цвета для отрисовки.
PALETTE: dict[str, tuple[int, int, int]] = {
    "person": (90, 205, 90),
    "laptop": (255, 160, 60),
    "mouse": (0, 165, 255),
    "keyboard": (0, 140, 255),
    "cell phone": (0, 120, 255),
    "potted plant": (215, 90, 220),
    "vase": (215, 90, 220),
    "tv": (180, 180, 90),
    "book": (140, 140, 240),
    "cup": (80, 200, 240),
    "bottle": (80, 220, 220),
}

FALLBACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (200, 200, 200),
    (160, 210, 255),
    (255, 190, 160),
    (180, 230, 180),
    (230, 180, 230),
    (190, 190, 250),
)


def class_color(name: str, class_id: int = 0) -> tuple[int, int, int]:
    """Стабильный цвет рамки для класса объекта."""
    if name in PALETTE:
        return PALETTE[name]
    return FALLBACK_COLORS[class_id % len(FALLBACK_COLORS)]
