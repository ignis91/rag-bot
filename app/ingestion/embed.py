import logging
import json
import psycopg
import sys

from app.core.log_config import setup_logging
from sentence_transformers import SentenceTransformer
from app.core.config import MODEL_NAME, DATABASE_URL, CHUNKS_DIR
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)


def embed_file(
    chunk_path: str, model: SentenceTransformer, batch_size: int, conn
) -> None:

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    if not chunks:
        logger.warning("Chunks %s is empty!", chunk_path)
        return

    video_id = chunks[0]["video_id"]

    with conn.cursor() as curr:
        curr.execute(
            "SELECT EXISTS(SELECT 1 FROM embeddings WHERE video_id = %s);", (video_id,)
        )
        already_exists = curr.fetchone()[0]

    if already_exists:
        logger.info("Embeddings for this file %s exists, skipping", video_id)
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=batch_size)

    with conn.transaction():
        with conn.cursor() as curr:
            for chunk, embedding in zip(chunks, embeddings):
                curr.execute(
                    """
                    INSERT INTO embeddings(
                        chunk_id,
                        video_id,
                        text,
                        chunk_start,
                        chunk_end,
                        segment_start_idx,
                        segment_end_idx,
                        embedding
                    ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["video_id"],
                        chunk["text"],
                        chunk["chunk_start"],
                        chunk["chunk_end"],
                        chunk["segment_start_idx"],
                        chunk["segment_end_idx"],
                        embedding,
                    ),
                )


def main() -> None:
    setup_logging()

    if not CHUNKS_DIR.exists():
        logger.error("Chunks directory doesn't exist")
        sys.exit(1)

    batch_size = 8
    model = SentenceTransformer(MODEL_NAME)

    ok = 0
    total = 0
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        register_vector(conn)
        for chunk_path in sorted(CHUNKS_DIR.glob("*.json")):
            total += 1
            try:
                embed_file(chunk_path, model, batch_size, conn)
                ok += 1
            except Exception as e:
                logger.error("Embedding error %s: %s", chunk_path, e)
    logger.info("Done: %d/%d", ok, total)


if __name__ == "__main__":
    main()
