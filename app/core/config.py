from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LECTURES_DIR = BASE_DIR / "data" / "lectures"
TRANSCRIPTS_DIR = BASE_DIR / "data" / "transcripts"
LOG_DIR = BASE_DIR / "logs"
TARGET_TOKENS = 512
MODEL_NAME = "BAAI/bge-m3"
