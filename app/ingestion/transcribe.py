import sys
import ctypes
from pathlib import Path

# Жесткое динамическое подключение библиотек CUDA из site-packages виртуального окружения
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib

    cublas_path = list(nvidia.cublas.lib.__path__)[0]
    cudnn_path = list(nvidia.cudnn.lib.__path__)[0]

    # Принудительно загружаем .so файлы в память процесса
    ctypes.CDLL(str(Path(cublas_path) / "libcublas.so.12"))
    ctypes.CDLL(
        str(Path(cudnn_path) / "libcubnn.so.9")
    )  # или libcudnn.so.8 в зависимости от версии
except Exception as e:
    # Если на сервере настроена системная CUDA, этот шаг пропустится
    pass


from faster_whisper import WhisperModel
import json

# 1. Вычисляем корень проекта (RAG-BOT) относительно этого файла
# __file__ -> app/ingestion/transcribe.py
# .parent -> app/ingestion/
# .parent.parent -> корень проекта RAG-BOT/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Настройка путей для структуры транскрибирования
LECTURES_DIR = BASE_DIR / "data" / "lectures"
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"

# Выбор модели faster whisper
model_size = "small"
model = WhisperModel(model_size, device="cuda", compute_type="int8")


# Разделение на сегменты через faster whisper
video_path = LECTURES_DIR / "video1.mp4"
segments, info = model.transcribe(str(video_path), beam_size=1)

# Реальная транскрипция в связной список
result_segments = []
for segment in segments:
    result_segments.append(
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text,
        }
    )

# Запись результатов сегмента в json формат в папку transcripts
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

out_file = TRANSCRIPTS_DIR / "video1.json"
data = {
    "video_id": "video1",
    "model": model_size,
    "language": info.language,
    "segments": result_segments,
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
