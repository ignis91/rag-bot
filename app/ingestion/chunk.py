from app.core.config import MODEL_NAME, TRANSCRIPTS_DIR, TARGET_TOKENS
from transformers import AutoTokenizer
import logging
import json


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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transcribe_files = list(TRANSCRIPTS_DIR.glob("*.json"))

    for transcribe_path in transcribe_files:
        transcript = json.loads(transcribe_path.read_text(encoding="utf-8"))
        chunks = chunk_transcript(transcript, tokenizer, TARGET_TOKENS)
        print(len(chunks))
        print(count_tokens(chunks[0]["text"], tokenizer))


if __name__ == "__main__":
    main()
