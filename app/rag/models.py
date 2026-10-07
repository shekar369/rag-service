"""Model + vector store factories. Swap providers via env only.

LLM_PROVIDER / EMBEDDING_PROVIDER values:
  nvidia        : NVIDIA API Catalog (build.nvidia.com) via langchain-nvidia-ai-endpoints
  openai_compat : any OpenAI-compatible server, e.g. local Ollama at http://localhost:11434/v1
  ollama        : native Ollama client (langchain-ollama)
  huggingface   : local sentence-transformers (embeddings only)
  anthropic | bedrock_converse | openai | ... : via init_chat_model (LLM only)
"""
import logging
from functools import lru_cache

from langchain_chroma import Chroma

from app.config import settings

log = logging.getLogger(__name__)


def _require_nvidia_key() -> str:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env, it should start with 'nvapi-').")
    return settings.nvidia_api_key


def validate_nvidia_models() -> None:
    """Fail fast at startup if a configured NVIDIA model was retired or misspelled.

    The hosted catalog changes often (models get end-of-life dates), and a retired
    model otherwise only surfaces as a 410 on the first chat request.
    """
    wanted = {}
    if settings.llm_provider == "nvidia":
        wanted["LLM_MODEL"] = settings.llm_model
    if settings.embedding_provider == "nvidia":
        wanted["EMBEDDING_MODEL"] = settings.embedding_model
    if not wanted:
        return

    import httpx

    try:
        r = httpx.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {_require_nvidia_key()}"},
            timeout=15,
        )
        r.raise_for_status()
        ids = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:  # network hiccup: don't block startup
        log.warning("could not verify NVIDIA model list: %s", e)
        return
    if not ids:
        return

    if "LLM_MODEL" in wanted and wanted["LLM_MODEL"] not in ids:
        raise RuntimeError(
            f"LLM_MODEL '{wanted['LLM_MODEL']}' is not in the NVIDIA catalog for your key "
            "(retired or misspelled). List current models: GET https://integrate.api.nvidia.com/v1/models"
        )
    if "EMBEDDING_MODEL" in wanted and wanted["EMBEDDING_MODEL"] not in ids:
        log.error(
            "EMBEDDING_MODEL '%s' was not found in the model list; ingestion/search may fail",
            wanted["EMBEDDING_MODEL"],
        )


def get_llm(**kwargs):
    p = settings.llm_provider

    if p == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=settings.llm_model,          # e.g. meta/llama-3.3-70b-instruct (verify in catalog)
            api_key=_require_nvidia_key(),
            temperature=0,
            **kwargs,
        )

    if p == "openai_compat":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            stream_usage=True,
            **kwargs,
        )

    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
            reasoning=False,
            **kwargs,
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(settings.llm_model, model_provider=p, temperature=0, **kwargs)


@lru_cache
def get_embeddings():
    p = settings.embedding_provider

    if p == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        # NVIDIAEmbeddings sets input_type (query vs passage) per call, which NVIDIA's
        # retrieval embedding models require. truncate="END" avoids errors on long chunks.
        return NVIDIAEmbeddings(
            model=settings.embedding_model,    # e.g. nvidia/nv-embedqa-e5-v5 (verify in catalog)
            api_key=_require_nvidia_key(),
            truncate="END",
        )
    if p == "openai_compat":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            check_embedding_ctx_length=False,
        )
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    if p == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=settings.embedding_model)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )