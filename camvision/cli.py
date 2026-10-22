"""Командная строка CameraVision."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import logging
import platform
import sys
import time
from pathlib import Path

from . import __version__

import cv2

from .config import COCO_RU, LABELS_PATH, MODELS_DIR, SNAPSHOT_DIR
from .detector import DetectorError, YoloDetector
from .drawing import Painter, draw_scene
from .faces import FaceDetector
from .finegrained import ClipClassifier, LabelBook
from .output import JsonlWriter, VideoRecorder, save_snapshot, timestamp
from .pipeline import Analyzer, AnalyzerOptions, FrameResult
from .sources import VideoSource, is_image_source, load_image, probe_cameras

log = logging.getLogger("camvision")
WINDOW = "CameraVision"


# ----------------------------------------------------------------------
def setup_console(quiet: bool, verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_source(value: str) -> int | str:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="Распознавание объектов, людей и предположительных моделей/видов по камере.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("Источник")
    src.add_argument("--source", default="0", help="камера (0, 1, …), файл или rtsp/http-поток")
    src.add_argument("--image", help="обработать одно изображение (путь или URL)")
    src.add_argument("--width", type=int, help="ширина кадра камеры")
    src.add_argument("--height", type=int, help="высота кадра камеры")
    src.add_argument("--fps", type=float, help="запросить FPS камеры")
    src.add_argument("--fourcc", help="кодек камеры, например MJPG")
    src.add_argument("--loop", action="store_true", help="зациклить видеофайл")
    src.add_argument("--mirror", action="store_true", help="зеркалить кадр (как в селфи)")

    det = parser.add_argument_group("Детектор")
    det.add_argument("--model", default="yolo11n.pt", help="веса YOLO (.pt/.onnx) или имя модели")
    det.add_argument("--imgsz", type=int, default=640, help="размер входа детектора")
    det.add_argument("--conf", type=float, default=0.35, help="порог уверенности")
    det.add_argument("--iou", type=float, default=0.5, help="порог NMS")
    det.add_argument("--device", help="cpu, cuda, 0 … (по умолчанию авто)")
    det.add_argument("--classes", nargs="*", help="оставить только классы (имена или id)")
    det.add_argument("--max-det", type=int, default=40, help="максимум объектов на кадр")
    det.add_argument("--detect-every", type=int, default=1, help="запускать детекцию раз в N кадров")

    fg = parser.add_argument_group("Модели и виды (CLIP)")
    fg.add_argument("--no-fine-grained", action="store_true", help="выключить определение моделей/видов")
    fg.add_argument("--fg-model", default="ViT-B-32", help="модель open_clip")
    fg.add_argument("--fg-pretrained", default="laion2b_s34b_b79k", help="веса open_clip")
    fg.add_argument("--fg-topk", type=int, default=3, help="сколько гипотез показывать")
    fg.add_argument("--fg-samples", type=int, default=4, help="кадров на один объект для CLIP")
    fg.add_argument("--fg-refresh", type=float, default=6.0, help="пересчёт гипотез, секунд")
    fg.add_argument("--fg-margin", type=float, help="порог «уверенности» (перекрывает JSON)")
    fg.add_argument("--labels", default=str(LABELS_PATH), help="JSON со списками моделей/видов")

    people = parser.add_argument_group("Люди")
    people.add_argument("--no-faces", action="store_true", help="не искать лица")
    people.add_argument("--face-every", type=int, default=3, help="искать лица раз в N кадров")

    out = parser.add_argument_group("Вывод")
    out.add_argument("--no-display", "--headless", dest="headless", action="store_true", help="без окна")
    out.add_argument("--record", help="записать аннотированное видео в mp4")
    out.add_argument("--jsonl", help="писать результаты в JSONL")
    out.add_argument("--jsonl-every", type=int, default=1, help="каждый N-й кадр в JSONL")
    out.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR), help="куда сохранять снимки")
    out.add_argument("--save", help="куда сохранить размеченную картинку (режим --image)")
    out.add_argument("--frames", type=int, help="остановиться после N кадров")
    out.add_argument("--benchmark", type=int, metavar="N", help="обработать N кадров без окна и показать FPS")
    out.add_argument("--quiet", action="store_true", help="меньше вывода")
    out.add_argument("--verbose", action="store_true", help="подробный лог")

    info = parser.add_argument_group("Диагностика")
    info.add_argument("--doctor", action="store_true", help="проверить окружение и камеры")
    info.add_argument("--list-cameras", action="store_true", help="найти доступные камеры")
    info.add_argument("--list-classes", action="store_true", help="показать классы COCO")
    info.add_argument("--version", action="version", version=f"CameraVision {__version__}")
    return parser


# ----------------------------------------------------------------------
def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "не установлен"


def doctor() -> int:
    print(f"CameraVision {__version__}")
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"ОС          : {platform.platform()}")
    print(f"opencv      : {cv2.__version__}")
    print(f"numpy       : {_version('numpy')}")
    print(f"ultralytics : {_version('ultralytics')}")
    print(f"torch       : {_version('torch')}")
    print(f"open_clip   : {_version('open_clip_torch')}")
    print(f"pillow      : {_version('Pillow')}")

    painter = Painter()
    print(f"Шрифт RU    : {painter.regular or 'не найден (будет латиница)'}")

    try:
        import torch

        print(f"CUDA        : {'да' if torch.cuda.is_available() else 'нет (CPU)'}")
    except Exception:
        print("CUDA        : torch недоступен")

    try:
        book = LabelBook(LABELS_PATH)
        summary = ", ".join(f"{g.key}: {len(g.labels)}" for g in book.groups)
        print(f"Подсказки   : {summary or 'нет'}")
    except Exception as exc:
        print(f"Подсказки   : ошибка загрузки — {exc}")

    print("\nКамеры:")
    found = probe_cameras()
    if not found:
        print("  не найдено (закрыты другим приложением?)")
    for index, width, height in found:
        print(f"  #{index}: {width}x{height}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = sorted(p.name for p in MODELS_DIR.glob("*") if p.suffix in {".pt", ".onnx", ".engine"})
    print("\nВеса в models/: " + (", ".join(weights) if weights else "нет (скачаются при первом запуске)"))
    print("\nПроверка завершена.")
    return 0


def list_cameras() -> int:
    found = probe_cameras()
    if not found:
        print("Камеры не найдены. Проверьте разрешения Windows и закройте Zoom/Teams.")
        return 1
    print("Доступные камеры:")
    for index, width, height in found:
        print(f"  --source {index}   ({width}x{height})")
    return 0


def list_classes() -> int:
    print(f"{'id':>3}  {'english':<20} русский")
    for index, (name, ru) in enumerate(COCO_RU.items()):
        print(f"{index:>3}  {name:<20} {ru}")
    return 0


# ----------------------------------------------------------------------
def _make_analyzer(args: argparse.Namespace, *, image_mode: bool = False) -> Analyzer:
    labels = LabelBook(args.labels)
    if len(labels) == 0:
        log.warning("Списки моделей пусты: %s", args.labels)

    detector = YoloDetector(
        args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
    ).load()

    if args.classes:
        name_to_id = {v: k for k, v in detector.names.items()}
        ids: list[int] = []
        for item in args.classes:
            if str(item).isdigit():
                ids.append(int(item))
            elif item in name_to_id:
                ids.append(name_to_id[item])
            elif item in COCO_RU.values():
                reverse = {ru: en for en, ru in COCO_RU.items()}
                en = reverse.get(item)
                if en in name_to_id:
                    ids.append(name_to_id[en])
                else:
                    log.warning("Класс не найден: %s", item)
            else:
                log.warning("Класс не найден: %s", item)
        detector.classes = sorted(set(ids)) or None

    classifier = None
    options = AnalyzerOptions(
        fg_enabled=not args.no_fine_grained,
        fg_topk=args.fg_topk,
        fg_samples=max(1, args.fg_samples),
        fg_refresh_s=args.fg_refresh,
        fg_margin=args.fg_margin,
        faces_enabled=not args.no_faces,
        face_every=max(1, args.face_every),
    )
    if image_mode:
        options.fg_samples = 1
        options.fg_sample_gap_s = 0.0

    if not args.no_fine_grained:
        classifier = ClipClassifier(
            args.fg_model,
            args.fg_pretrained,
            device=args.device,
        )
        if classifier.ensure_loaded():
            log.info("CLIP готов: %s (%s)", classifier.short_name, classifier.device)
        else:
            log.warning("CLIP недоступен — для растений останутся цветовые подсказки.")

    face_detector = None if args.no_faces else FaceDetector()
    if face_detector is not None and not face_detector.ok:
        log.warning("Каскад лиц не найден в OpenCV — поиск лиц отключён.")
        face_detector = None

    return Analyzer(
        detector,
        labels,
        classifier=classifier,
        face_detector=face_detector,
        options=options,
    )


def _print_events(result: FrameResult) -> None:
    for kind, obj in result.events:
        if kind == "new":
            print(f"  + #{obj.track_id} {obj.ru} ({obj.conf * 100:.0f}%)")
        elif kind == "fine":
            guesses = " · ".join(f"{h.name} {h.score * 100:.0f}%" for h in obj.hypotheses[:3])
            prefix = "не уверен: " if obj.uncertain else ""
            print(f"      ↳ #{obj.track_id} {obj.ru}: {prefix}{guesses}")
            if obj.clothing and obj.name == "person":
                print(f"        одежда: {obj.clothing}")


def _print_result_table(result: FrameResult) -> None:
    print(f"\nНайдено объектов: {len(result.objects)} · людей: {result.person_count}\n")
    header = f"{'':<4}{'объект':<24}{'уверен.':>8}  предположение"
    print(header)
    print("-" * max(60, len(header)))
    for obj in result.objects:
        if obj.hypotheses:
            guesses = " · ".join(
                f"{h.name} {h.score * 100:.0f}%" for h in obj.hypotheses[:3]
            )
            if obj.uncertain:
                guesses = "не уверен · " + guesses
        else:
            guesses = "—"
        extra = ""
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
            extra = "  [" + ", ".join(bits) + "]" if bits else ""
        print(f"#{obj.track_id:<3}{obj.ru:<24}{obj.conf:>8.2f}  {guesses}{extra}")
    print()


def _watermark(result: FrameResult, frame_w: int, frame_h: int, fg_enabled: bool) -> str:
    parts = [f"FPS {result.fps:4.1f}", result.detector_name or "YOLO"]
    if fg_enabled and result.classifier_name:
        parts.append(result.classifier_name)
    elif not fg_enabled:
        parts.append("без CLIP")
    parts.append(f"{result.infer_ms:.0f}+{result.fine_ms:.0f} ms")
    parts.append(f"{frame_w}x{frame_h}")
    return " | ".join(parts)


def _screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1280, 720


def _imshow_fit(frame) -> None:
    h, w = frame.shape[:2]
    sw, sh = _screen_size()
    max_w = int(sw * 0.88)
    max_h = int(sh * 0.82)
    scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
    shown = frame if scale >= 1.0 else cv2.resize(frame, None, fx=scale, fy=scale)
    try:
        cv2.imshow(WINDOW, shown)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV собран без поддержки окон. Запустите с --no-display "
            "или установите opencv-python (не headless)."
        ) from exc


def _window_alive() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return True


# ----------------------------------------------------------------------
def run_image(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    analyzer = _make_analyzer(args, image_mode=True)
    painter = Painter()

    result = analyzer.process(image, detect=True)
    # Второй проход даёт лицам/одежде/гипотезам «дожаться» на одном кадре.
    result = analyzer.process(image, detect=True)
    if result.objects and any(o.hypotheses for o in result.objects):
        result = analyzer.process(image, detect=True)

    fg_on = not args.no_fine_grained
    h, w = image.shape[:2]
    annotated = draw_scene(
        image,
        result,
        painter,
        show_fine=fg_on,
        show_panel=True,
        show_hints=False,
        watermark=_watermark(result, w, h, fg_on),
    )
    _print_events(result)
    _print_result_table(result)

    target = args.save or (Path(args.snapshot_dir) / f"detected_{timestamp()}.jpg")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        target.write_bytes(encoded.tobytes())
        print(f"Размеченное изображение: {target}")

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        _imshow_fit(annotated)
        print("Нажмите любую клавишу в окне, чтобы выйти.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(args: argparse.Namespace) -> int:
    source = VideoSource(
        parse_source(args.source),
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        loop=args.loop,
    )
    source.open()
    analyzer = _make_analyzer(args)
    painter = Painter()

    recorder: VideoRecorder | None = None
    if args.record:
        w, h = source.frame_size
        recorder = VideoRecorder(args.record, source.fps, (w or 1280, h or 720))
    jsonl = JsonlWriter(args.jsonl, every=args.jsonl_every) if args.jsonl else None

    fg_on = not args.no_fine_grained
    show_panel = not args.headless
    mirror = args.mirror
    paused = False
    processed = 0
    started = time.perf_counter()
    last_annotated = None
    limit = args.benchmark or args.frames

    print(
        f"Источник: {source.describe()} · детектор: {analyzer.detector.describe()} · "
        f"CLIP: {analyzer.classifier.short_name if analyzer.classifier else 'выкл'}"
    )
    if not args.headless:
        print("Клавиши: q — выход, s — снимок, f — CLIP, h — панель, m — зеркало, пробел — пауза")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            detect = (processed % max(1, args.detect_every)) == 0
            result = analyzer.process(frame, detect=detect)
            fg_active = fg_on and analyzer.fg_enabled
            h, w = frame.shape[:2]
            annotated = draw_scene(
                frame,
                result,
                painter,
                show_panel=show_panel,
                show_fine=fg_active,
                show_faces=not args.no_faces,
                show_hints=not args.headless,
                watermark=_watermark(result, w, h, fg_active),
            )
            last_annotated = annotated
            processed += 1

            if not args.quiet:
                _print_events(result)
            if jsonl is not None:
                jsonl.write(result, source=source.describe())
            if recorder is not None:
                recorder.write(annotated)

            if args.headless:
                if processed % 30 == 0 and not args.quiet:
                    print(
                        f"кадр {processed}: {len(result.objects)} объектов, "
                        f"{result.fps:.1f} FPS, {result.infer_ms + result.fine_ms:.0f} мс"
                    )
            else:
                _imshow_fit(annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = save_snapshot(annotated, args.snapshot_dir)
                    print(f"Снимок: {path}")
                elif key == ord("f"):
                    fg_on = not fg_on
                    analyzer.fg_enabled = fg_on
                    print(f"CLIP: {'вкл' if fg_on else 'выкл'}")
                elif key == ord("h"):
                    show_panel = not show_panel
                elif key == ord("m"):
                    mirror = not mirror
                    print(f"Зеркало: {'вкл' if mirror else 'выкл'}")
                elif key == ord("r"):
                    analyzer.reset()
                    print("Трекер сброшен")
                elif key == ord(" "):
                    paused = True
                    while paused:
                        if last_annotated is not None:
                            _imshow_fit(last_annotated)
                        k = cv2.waitKey(30) & 0xFF
                        if k == ord(" "):
                            paused = False
                        elif k in (ord("q"), 27):
                            return 0
                        elif not _window_alive():
                            return 0
                if not _window_alive():
                    break

            if limit and processed >= limit:
                break
    finally:
        elapsed = time.perf_counter() - started
        source.release()
        if recorder is not None:
            recorder.close()
        if jsonl is not None:
            jsonl.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = max(1e-6, time.perf_counter() - started)
    print(
        f"\nОбработано кадров: {processed} за {elapsed:.1f} с "
        f"({processed / elapsed:.1f} FPS в среднем)"
    )
    if args.record:
        print(f"Видео: {args.record}")
    if args.jsonl:
        print(f"JSONL: {args.jsonl}")
    return 0


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_console(args.quiet, args.verbose)

    if args.doctor:
        return doctor()
    if args.list_cameras:
        return list_cameras()
    if args.list_classes:
        return list_classes()

    if args.benchmark:
        args.headless = True
        args.quiet = False

    try:
        if args.image:
            return run_image(args)
        if isinstance(parse_source(args.source), str) and is_image_source(str(args.source)):
            args.image = str(args.source)
            return run_image(args)
        return run_stream(args)
    except DetectorError as exc:
        log.error("\n%s", exc)
        return 2
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        return 130
    except Exception as exc:
        if args.verbose:
            log.exception("Ошибка")
        else:
            log.error("Ошибка: %s", exc)
            log.error("Подробности: --verbose")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
