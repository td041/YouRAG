"""
LLM Client tối giản - hỗ trợ Groq, OpenAI, Gemini, Ollama.

Thiết kế:
- Provider pattern: dễ swap giữa Groq ↔ OpenAI ↔ Gemini ↔ Ollama
- Chỉ expose 1 method chính: chat_complete()
- Groq rate limit → tự động fallback sang Gemini nếu có GEMINI_API_KEY
- Timeout rõ ràng, retry đơn giản
"""

import os
import re
import time
from typing import Optional

from src.core.config import settings
from src.core.logger import logger


_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMClient:
    """Giao tiếp với LLM provider (Groq / OpenAI / Gemini / Ollama).

    Ưu tiên tự động:
      1. Groq   → nhanh nhất, free tier rộng (llama-3.3-70b)
      2. Gemini → fallback khi Groq rate limit (gemini-1.5-flash, free)
      3. OpenAI → nếu set LLM_PROVIDER=openai
      4. Ollama → local, hoàn toàn free
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

        # 1. Chọn Provider theo độ ưu tiên: Tham số truyền vào > config .env > Auto-detect
        env_provider = getattr(settings, "LLM_PROVIDER", "groq").lower()

        # Mở khóa SecretStr thành chuỗi thật
        groq_key = settings.GROQ_API_KEY.get_secret_value() if getattr(settings, "GROQ_API_KEY", None) else os.getenv("GROQ_API_KEY", "")
        openai_key = settings.OPENAI_API_KEY.get_secret_value() if getattr(settings, "OPENAI_API_KEY", None) else os.getenv("OPENAI_API_KEY", "")
        gemini_key = settings.GEMINI_API_KEY.get_secret_value() if getattr(settings, "GEMINI_API_KEY", None) else os.getenv("GEMINI_API_KEY", "")

        if provider:
            self.provider = provider
        elif env_provider == "ollama":
            self.provider = "ollama"
        elif env_provider == "openai":
            self.provider = "openai"
        elif env_provider == "gemini":
            self.provider = "gemini"
        else:
            self.provider = "groq"  # Mặc định

        # Lưu Gemini key để dùng cho auto-fallback
        self._gemini_key = gemini_key
        self._fallback_client: Optional[object] = None

        # 2. Khởi tạo Client tương ứng
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=groq_key)
            self.model = model or getattr(settings, "LLM_CONTEXTUAL_MODEL", "llama-3.1-8b-instant")
            logger.info(f"LLMClient → Groq | model={self.model}")

            # Chuẩn bị Gemini fallback nếu có key
            if gemini_key:
                from openai import OpenAI
                self._fallback_client = OpenAI(api_key=gemini_key, base_url=_GEMINI_BASE_URL)
                self._fallback_model = "gemini-1.5-flash"
                logger.info("LLMClient → Gemini fallback ready")

        elif self.provider == "gemini":
            from openai import OpenAI
            self._client = OpenAI(api_key=gemini_key, base_url=_GEMINI_BASE_URL)
            self.model = model or "gemini-1.5-flash"
            logger.info(f"LLMClient → Gemini | model={self.model}")

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
            raise ValueError(f"Provider không hỗ trợ: {self.provider}")

    def chat_complete(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Gọi LLM, trả về nội dung text. Có retry với rate-limit-aware backoff."""
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

                # Parse wait time từ nhiều format khác nhau của Groq/OpenAI
                wait_time = 2.0 * attempt  # default exponential backoff
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
                    logger.warning(f"Rate limit (attempt {attempt}/{self.max_retries}) → chờ {wait_time:.1f}s")
                    # Auto-fallback sang Gemini nếu Groq rate limit và có fallback
                    if self._fallback_client is not None:
                        logger.info("Groq rate limit → fallback sang Gemini")
                        try:
                            resp = self._fallback_client.chat.completions.create(
                                model=self._fallback_model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                            return resp.choices[0].message.content.strip()
                        except Exception as fe:
                            logger.warning(f"Gemini fallback cũng lỗi: {fe}")
                else:
                    logger.warning(f"LLM attempt {attempt}/{self.max_retries} lỗi: {e}")

                if attempt < self.max_retries:
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM call thất bại sau {self.max_retries} lần: {e}") from e

        return ""

    def chat_complete_stream(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """Gọi LLM và Yield từng chunk (Streaming)."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        # Chọn model phù hợp: Dùng model lớn nếu là Groq, còn không thì dùng model mặc định của provider
        if self.provider == "groq":
            target_model = getattr(settings, "LLM_MODEL_NAME", "llama-3.3-70b-versatile")
        else:
            target_model = self.model

        def _stream_from(client: object, mdl: str):
            stream = client.chat.completions.create(  # type: ignore[attr-defined]
                model=mdl,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        try:
            yield from _stream_from(self._client, target_model)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and self._fallback_client is not None:
                logger.warning("Groq rate limit (stream) → fallback sang Gemini")
                try:
                    yield from _stream_from(self._fallback_client, self._fallback_model)
                    return
                except Exception as fe:
                    logger.warning(f"Gemini stream fallback lỗi: {fe}")
            logger.error(f"Lỗi Streaming LLM: {e}")
            yield f"Lỗi Streaming: {str(e)}"
