from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings reads .env into Settings only; it does NOT export to os.environ.
# LangSmith (LANGSMITH_*), HF_HUB_OFFLINE and provider SDK keys read os.environ,
# so load .env into the process environment as well.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM: local Ollama via its OpenAI-compatible API ----
    llm_provider: str = "nvidia"   # nvidia | openai_compat | ollama | anthropic | openai ...
    llm_model: str = "qwen3.5:4b"         # must match `ollama list`
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"           # placeholder; Ollama ignores it

    # Read from NVIDIA_API_KEY in .env (never hardcode keys in this file)
    nvidia_api_key: str | None = None

    # Only used when llm_provider == "ollama" (native client)
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192

    # ---- Embeddings ----
    embedding_provider: str = "huggingface"  # nvidia | huggingface | openai_compat | ollama | openai | bedrock
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ---- Storage ----
    chroma_dir: str = "./.chroma"
    collection: str = "docs"
    data_dir: str = "./data"

    # ---- RAG tuning ----
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 6
    max_history_turns: int = 6

    # Cloud keys never go in this file. To use OpenAI instead, put in .env:
    #   LLM_PROVIDER=openai
    #   LLM_MODEL=<model>
    #   OPENAI_API_KEY=<new key>


settings = Settings()