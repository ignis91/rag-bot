import logging
import json
from app.core.log_config import setup_logging
from sentence_transformers import SentenceTransformer
from app.core.config import MODEL_NAME

logger = logging.getLogger(__name__)


def embed_file(
    chunk_path: str, model: SentenceTransformer, batch_size: int, conn
) -> None:

    with open(chunk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_id = ...  # TODO Достать video_id из data

    # TODO: idempotency — SQL-запрос к БД: существуют ли строки для этого video_id
    if ...:
        return

    # TODO Прогнать SentenceTransformer из аргумента эмбеддинг,  запись в pgvector — транзакцией (all-or-nothing для video_id)
    texts = ...  # Достать текст из файла data
    embeddings = model.encode(texts, batch_size=batch_size)


# TODO запись в pgvector — транзакцией (all-or-nothing для video_id)


def main() -> None:
    setup_logging()
    model = SentenceTransformer(MODEL_NAME)
    conn = ...  # открыть подключение к Postgres

    # TODO ?Проверка существования бд для записи в pjvector

    chunk_files = ...  # Достать файлы из CHUNKS_DIR
    batch_size = 8

    for chunk_path in chunk_files:
        try:
            embed_file(chunk_path, model, batch_size, conn)
        except Exception as e:
            logger.error(f"Embedding error {chunk_path}: {e}")
