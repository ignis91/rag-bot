import json
import sys
import logging
from faster_whisper import WhisperModel
from app.core.config import LECTURES_DIR, TRANSCRIPTS_DIR
from app.core.log_config import setup_logging

logger = logging.getLogger(__name__)


def transcribe_file(video_path, model, transcripts_dir) -> None:
    """transcribe lecture videos to JSON transcripts."""

    file_name = video_path.stem
    out_file = transcripts_dir / f"{file_name}.json"

    # the artifact on disk is the completion marker, not the log
    if out_file.exists():
        return

    logger.info(f"Start transcribing {file_name}")

    # lazy generator: corrupt-audio errors surface during iteration, not here
    segments, info = model.transcribe(
        str(video_path), beam_size=1, language="zh", vad_filter=True
    )

    # ms precision is noise for retrieval
    result_segments = []

    for segment in segments:
        result_segments.append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text,
            }
        )

    data = {
        "video_id": f"{file_name}",
        "language": info.language,
        "segments": result_segments,
    }

    # write-then-rename: a crash mid-write leaves no half-written transcript
    tmp_file = out_file.with_name(out_file.name + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_file.replace(out_file)
    return


if __name__ == "__main__":
    setup_logging()

    if not LECTURES_DIR.exists():
        logger.error("The folder containing the lecture materials doesn't exist")
        sys.exit(1)

    video_files = list(LECTURES_DIR.glob("*.mp4"))
    if not video_files:
        logger.warning("Lectures haven't been uploaded")
        sys.exit(0)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    model_size = "small"
    model = WhisperModel(model_size, device="cuda", compute_type="int8")

    for video_path in video_files:
        # error boundary per file: one bad video must not kill the run
        try:
            transcribe_file(video_path, model, TRANSCRIPTS_DIR)
        except Exception as e:
            logger.error(f"Processing error {video_path}: {e}")
