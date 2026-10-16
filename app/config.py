from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # A file path, not a URL. The store is SQLite on the pharmacy's own disk.
    database_path: str = "pharmacy.db"

    # The shop's own stock list. None means use the starter formulary.
    formulary_path: str | None = None

    ollama_url: str = "http://localhost:11434"
    gemma_model: str = "gemma3:4b"
    ollama_timeout_s: float = 600.0
    num_ctx: int = 4096
    # Safety net against runaway generation. At the measured 7.3 tok/s on this
    # CPU, every 100 tokens of output costs about 14 seconds of the
    # pharmacist's evening, so a page that wants more than this is a page we
    # would rather send to review than wait on.
    num_predict: int = 1400

    confidence_threshold: float = 0.8

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model: str = "eleven_multilingual_v2"

    sentry_dsn: str | None = None

    # Deterministic fixtures instead of calling Ollama. Set by the hosted demo,
    # which has no GPU and cannot hold the model.
    demo_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
