import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Centralized application settings and paths loaded from environment."""

    # Directories
    project_root: Path = Path(__file__).resolve().parents[2]
    docs_dir: Path = Path(os.getenv("DOCS_DIR", project_root / "data" / "docs"))
    chroma_persist_dir: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", project_root / "storage" / "chroma")
    )

    # Providers & Models
    # LLM provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name: str = os.getenv("MODEL_NAME", "llama3:instruct")

    # Embedding provider
    embed_provider: str = os.getenv("EMBED_PROVIDER", "huggingface").lower()
    embed_model: str = os.getenv(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # API
    port: int = int(os.getenv("PORT", "8000"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # CORS
    @property
    def cors_allow_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "*")
        if raw.strip() == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    cors_allow_methods: list[str] = [
        m.strip() for m in os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS").split(",")
    ]
    cors_allow_headers: list[str] = [
        h.strip() for h in os.getenv("CORS_ALLOW_HEADERS", "*").split(",")
    ]


settings = Settings()


