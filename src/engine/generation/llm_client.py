"""
LLM Client — supports Groq, OpenAI, Ollama.

- Provider pattern: easy swap between Groq ↔ OpenAI ↔ Ollama
- Groq backup key rotation when rate limited (GROQ_API_KEYS)
- Retry with rate-limit-aware backoff
"""

import os
import re
import time
from typing import Optional

from src.core.config import settings
from src.core.logger import logger


class LLMClient:
    """LLM provider client (Groq / OpenAI / Ollama).

    Priority:
      1. Groq   → fastest, free tier (llama-3.3-70b)
      2. OpenAI → if LLM_PROVIDER=openai
      3. Ollama → local, fully free
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries

        env_provider = getattr(settings, "LLM_PROVIDER", "groq").lower()

        groq_key = settings.GROQ_API_KEY.get_secret_value() if getattr(settings, "GROQ_API_KEY", None) else os.getenv("GROQ_API_KEY", "")
        openai_key = settings.OPENAI_API_KEY.get_secret_value() if getattr(settings, "OPENAI_API_KEY", None) else os.getenv("OPENAI_API_KEY", "")

        if provider:
            self.provider = provider
        elif env_provider == "ollama":
            self.provider = "ollama"
        elif env_provider == "openai":
            self.provider = "openai"
        else:
            self.provider = "groq"

        self._groq_backup_clients: list = []
        self._groq_key_index: int = 0

        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=groq_key)
            self.model = model or getattr(settings, "LLM_CONTEXTUAL_MODEL", "llama-3.1-8b-instant")
            logger.info(f"LLMClient → Groq | model={self.model}")

            extra_keys_str = getattr(settings, "GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEYS", "")
            extra_keys = [k.strip() for k in extra_keys_str.split(",") if k.strip() and k.strip() != groq_key]
            self._groq_backup_clients = [Groq(api_key=k) for k in extra_keys]
            if self._groq_backup_clients:
                logger.info(f"LLMClient → {len(self._groq_backup_clients)} Groq backup key(s) loaded")

        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=openai_key)
            self.model = model or getattr(settings, "LLM_MODEL_NAME", "gpt-4o-mini")
            logger.info(f"LLMClient → OpenAI | model={self.model}")

        elif self.provider == "ollama":
            from openai import OpenAI
            base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self._client = OpenAI(api_key="ollama", base_url=base_url)
            self.model = model or "qwen2.5:7b-instruct-q4_K_M"
            logger.info(f"LLMClient → Ollama (LOCAL) | model={self.model}")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat_complete(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Call LLM, return text. Retry with rate-limit-aware backoff."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str.lower()

                wait_time = 2.0 * attempt
                for pattern in [
                    r"try again in (\d+\.?\d*)s",
                    r"retry after (\d+\.?\d*)",
                    r"wait (\d+\.?\d*) second",
                ]:
                    m = re.search(pattern, err_str, re.IGNORECASE)
                    if m:
                        wait_time = float(m.group(1)) + 0.5
                        break

                if is_rate_limit:
                    logger.warning(f"Rate limit (attempt {attempt}/{self.max_retries}) → wait {wait_time:.1f}s")

                    # Try Groq backup keys
                    backup_clients = getattr(self, "_groq_backup_clients", [])
                    if backup_clients and self._groq_key_index < len(backup_clients):
                        backup = backup_clients[self._groq_key_index]
                        self._groq_key_index += 1
                        logger.info(f"Groq rate limit → trying backup key #{self._groq_key_index}")
                        try:
                            resp = backup.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                            return resp.choices[0].message.content.strip()
                        except Exception as be:
                            logger.warning(f"Backup key #{self._groq_key_index} failed: {be}")

                    logger.warning(f"All Groq keys rate limited → waiting {wait_time:.1f}s")
                else:
                    logger.warning(f"LLM attempt {attempt}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries:
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {e}") from e

        return ""

    def chat_complete_stream(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """Call LLM and yield chunks (streaming)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        target_model = getattr(settings, "LLM_MODEL_NAME", "llama-3.3-70b-versatile") if self.provider == "groq" else self.model

        try:
            stream = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Streaming LLM error: {e}")
            yield f"Lỗi Streaming: {str(e)}"
