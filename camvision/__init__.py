"""CameraVision — распознавание объектов, людей и моделей по видеопотоку."""

import os

# OpenCV иначе сыпет предупреждениями о недоступных камерах/бэкендах.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

__version__ = "1.0.0"
__all__ = ["__version__"]

