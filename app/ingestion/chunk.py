from app.core.config import MODEL_NAME, TRANSCRIPTS_DIR, TARGET_TOKENS, CHUNKS_DIR
from transformers import AutoTokenizer
import logging
import json
import sys
from pathlib import Path
from app.core.log_config import setup_logging


logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer) -> int:
    tokens_count = len(tokenizer.encode(text))
    return tokens_count


def chunk_transcript(transcript: dict, tokenizer, target_tokens: int) -> list[dict]:
    video_id = transcript["video_id"]
    segments = transcript["segments"]

    chunks = []
    buffer = []
    current_tokens = 0

    for idx, segment in enumerate(segments):
        seg_tokens = count_tokens(segment["text"], tokenizer=tokenizer)

        if not buffer:
            segment_start_idx = idx
        buffer.append(segment)
        current_tokens += seg_tokens

        # overshoot: сегмент добирается целиком, target_tokens — мягкий ориентир
        if current_tokens >= target_tokens:
            chunk = {
                "chunk_id": f"{video_id}:{segment_start_idx}",
                "video_id": video_id,
                "chunk_start": buffer[0]["start"],
                "chunk_end": buffer[-1]["end"],
                "segment_start_idx": segment_start_idx,
                "segment_end_idx": idx,
                "text": "\n".join(seg["text"] for seg in buffer),
            }
            chunks.append(chunk)
            current_tokens = 0
            buffer = []

    # хвост, не добравший порога, — последний чанк лекции
    if buffer:
        chunk = {
            "chunk_id": f"{video_id}:{segment_start_idx}",
            "video_id": video_id,
            "chunk_start": buffer[0]["start"],
            "chunk_end": buffer[-1]["end"],
            "segment_start_idx": segment_start_idx,
            "segment_end_idx": idx,
            "text": "\n".join(seg["text"] for seg in buffer),
        }
        chunks.append(chunk)

    return chunks


def chunk_file(
    transcript_path: Path, tokenizer, target_tokens: int, chunks_dir: Path
) -> bool:
    output_path = chunks_dir / transcript_path.name

    if output_path.exists():
        logger.info("Chunks exist, skipping: %s", output_path.name)
        return True

    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to read transcript %s: %s", transcript_path.name, e)
        return False

    try:
        chunks = chunk_transcript(transcript, tokenizer, target_tokens)
    except Exception as e:
        logger.exception("Chunking failed for %s: %s", transcript_path.name, e)
        return False

    if not chunks:
        logger.warning("Zero chunks produced: %s", transcript_path.name)
        return False

    try:
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.error("Failed to write chunks %s: %s", output_path.name, e)
        return False

    logger.info("Wrote %d chunks: %s", len(chunks), output_path.name)
    return True


def main() -> None:
    setup_logging()

    if not TRANSCRIPTS_DIR.exists():
        logger.error("Transcripts dir not found: %s", TRANSCRIPTS_DIR)
        sys.exit(1)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ok = 0
    total = 0
    for transcript_path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        total += 1
        if chunk_file(transcript_path, tokenizer, TARGET_TOKENS, CHUNKS_DIR):
            ok += 1

    logger.info("Done: %d/%d transcripts chunked", ok, total)


if __name__ == "__main__":
    main()
