import json
from pathlib import Path
from faster_whisper import WhisperModel

# 1. Вычисляем корень проекта (RAG-BOT) относительно этого файла
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Настройка путей для структуры транскрибирования
LECTURES_DIR = BASE_DIR / "data" / "lectures"
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"

# Выбор модели (Переключено на medium для точности китайского языка)
model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

# Разделение на сегменты через faster whisper
video_path = LECTURES_DIR / "video1.mp4"
segments, info = model.transcribe(
    str(video_path), beam_size=5, language="zh", vad_filter=True
)

# Редукция избыточности: сборка сегментов с округлением таймстампов до 2 знаков
result_segments = []
for segment in segments:
    result_segments.append(
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text,
        }
    )

# Запись результатов в json формат
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
