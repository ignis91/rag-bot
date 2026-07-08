import json
import logging
from pathlib import Path
from faster_whisper import WhisperModel

# --------Вычисляем корень проекта (RAG-BOT) относительно этого файла--------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 0. --------Настройка конфигурации логгера--------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Формат строки
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Труба 1: консоль
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

# Труба 2: файл
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
errfile = logging.FileHandler(LOG_DIR / "errors.log", mode="w", encoding="utf=8")
errfile.setLevel(logging.ERROR)
errfile.setFormatter(formatter)

# Присоединяем трубы к логгеру
logger.addHandler(console)
logger.addHandler(errfile)

# 1. --------Настройка путей для структуры транскрибирования--------
LECTURES_DIR = BASE_DIR / "data" / "lectures"

# Проверка существования папки с лекциями/материаллом
if not LECTURES_DIR.exists():
    logger.error("Папки с лекциями не существет")

TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. --------Выбор модели (Переключено на medium для точности китайского языка)--------
model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

# 3. --------Разделение на сегменты через faster whisper для каждого файла--------
for video_path in LECTURES_DIR.glob("*.mp4"):
    file_name = video_path.stem
    out_file = TRANSCRIPTS_DIR / f"{file_name}.json"

    # Проверка на существование файла
    if out_file.exists():
        continue

    logger.info(f"Начинаем транскрибировать {file_name}")
    try:
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
        data = {
            "video_id": f"{file_name}",
            "model": model_size,
            "language": info.language,
            "segments": result_segments,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при обработке {file_name}: {e}")
        continue
