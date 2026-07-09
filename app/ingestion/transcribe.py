import json
import sys
import logging
from pathlib import Path
from faster_whisper import WhisperModel
from app.core.config import LECTURES_DIR, TRANSCRIPTS_DIR
from app.core.log_config import setup_logging

# 0. --------Настройка конфигурации логгера--------
logger = logging.getLogger(__name__)


def transcribe_file(video_path, model, transcripts_dir):
    file_name = video_path.stem
    out_file = transcripts_dir / f"{file_name}.json"

    if out_file.exists():
        return False

    logger.info(f"Начинаем транскрибировать {file_name}")

    segments, info = model.transcribe(
        str(video_path), beam_size=1, language="zh", vad_filter=True
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
        "language": info.language,
        "segments": result_segments,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


if __name__ == "__main__":
    setup_logging()

    # Проверка существования папки с лекциями/материаллом
    if not LECTURES_DIR.exists():
        logger.error("Папки с лекциями не существует")
        sys.exit(1)

    video_files = list(LECTURES_DIR.glob("*.mp4"))
    if not video_files:
        logger.warning("Лекции не загружены")
        sys.exit(0)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    model_size = "small"
    model = WhisperModel(model_size, device="cuda", compute_type="int8")

    for video_path in video_files:
        try:
            transcribe_file(video_path, model, TRANSCRIPTS_DIR)
        except Exception as e:
            logger.error(f"Ошибка при обработке {video_path}: {e}")
