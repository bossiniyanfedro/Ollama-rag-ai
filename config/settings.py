import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Centralized application settings and paths loaded from environment."""

    # Directories
    project_root: Path = Path(__file__).resolve().parents[1]
    docs_dir: Path = Path(os.getenv("DOCS_DIR", project_root / "data" / "docs"))
    chroma_persist_dir: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", project_root / "index" / "vectorstore")
    )

    # Models
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # API
    port: int = int(os.getenv("PORT", "8000"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()


