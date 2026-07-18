from app.core.config import (
    MODEL_NAME,
    TRANSCRIPTS_DIR,
    TARGET_TOKENS,
    CHUNKS_DIR,
    OVERLAP_TOKENS,
)
from transformers import AutoTokenizer
import logging
import json
import sys
from pathlib import Path
from app.core.log_config import setup_logging


logger = logging.getLogger(__name__)


def _build_chunk(
    video_id: str,
    buffer: list,  # список кортежей (idx, segment)
    segment_start_idx: int,
    segment_end_idx: int,
) -> dict:
    return {
        "chunk_id": f"{video_id}:{segment_start_idx}",
        "video_id": video_id,
        "chunk_start": buffer[0][1]["start"],
        "chunk_end": buffer[-1][1]["end"],
        "segment_start_idx": segment_start_idx,
        "segment_end_idx": segment_end_idx,
        "text": "\n".join(seg["text"] for _, seg in buffer),
    }


def count_tokens(text: str, tokenizer) -> int:
    tokens_count = len(tokenizer.encode(text))
    return tokens_count


def chunk_transcript(
    transcript: dict,
    tokenizer,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    if overlap_tokens >= target_tokens:
        raise ValueError("Overlap tokens more than target tokens")

    video_id = transcript["video_id"]
    segments = transcript["segments"]

    chunks = []
    buffer = []
    current_tokens = 0

    for idx, segment in enumerate(segments):
        seg_tokens = count_tokens(segment["text"], tokenizer=tokenizer)

        if not buffer:
            segment_start_idx = idx
        buffer.append((idx, segment))
        current_tokens += seg_tokens

        # overshoot: сегмент добирается целиком, target_tokens — мягкий ориентир
        if current_tokens >= target_tokens:
            overlap_sum = 0
            chunks.append(_build_chunk(video_id, buffer, segment_start_idx, idx))
            for j in range(len(buffer) - 1, -1, -1):
                overlap_sum += count_tokens(buffer[j][1]["text"], tokenizer=tokenizer)
                if overlap_sum >= overlap_tokens:
                    i = j
                    break
            buffer = buffer[i:]
            segment_start_idx = buffer[0][0]
            current_tokens = overlap_sum

    # хвост, не добравший порога, — последний чанк лекции
    if buffer:
        chunks.append(_build_chunk(video_id, buffer, segment_start_idx, buffer[-1][0]))

    return chunks


def chunk_file(
    transcript_path: Path,
    tokenizer,
    target_tokens: int,
    chunks_dir: Path,
    overlap_tokens: int,
) -> bool:
    final_path = chunks_dir / transcript_path.name

    if final_path.exists():
        logger.info("Chunks exist, skipping: %s", final_path.name)
        return True

    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to read transcript %s: %s", transcript_path.name, e)
        return False

    try:
        chunks = chunk_transcript(transcript, tokenizer, target_tokens, overlap_tokens)
    except Exception as e:
        logger.exception("Chunking failed for %s: %s", transcript_path.name, e)
        return False

    if not chunks:
        logger.warning("Zero chunks produced: %s", transcript_path.name)
        return False

    tmp_path = chunks_dir / f"{transcript_path.name}.tmp"

    try:
        tmp_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.error("Failed to write chunks %s: %s", final_path.name, e)
        return False

    tmp_path.replace(final_path)

    logger.info("Wrote %d chunks: %s", len(chunks), final_path.name)
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
        if chunk_file(
            transcript_path, tokenizer, TARGET_TOKENS, CHUNKS_DIR, OVERLAP_TOKENS
        ):
            ok += 1

    logger.info("Done: %d/%d transcripts chunked", ok, total)


if __name__ == "__main__":
    main()
