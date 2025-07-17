from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_CODE = BASE_DIR / "demo-source-code" # This is where the source code will be stored
SOURCE_CODE.mkdir(parents=True, exist_ok=True)

DEV_MODEL = "devstral-small-2505"
CODE_MODEL = "mistral-medium-latest"
EMBEDDING_MODEL = "codestral-embed"

RELEVANCE_THRESHOLD = 0.8
