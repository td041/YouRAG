"""
LLM Client tối giản - hỗ trợ Groq (Llama3) và OpenAI.

Thiết kế:
- Provider pattern: dễ swap giữa Groq ↔ OpenAI ↔ Ollama sau này
- Chỉ expose 1 method chính: chat_complete()
- Timeout rõ ràng, retry đơn giản
"""

import os
import re
import time
from typing import Optional

from src.core.config import settings
from src.core.logger import logger


class LLMClient:
    """Giao tiếp với LLM provider (Groq / OpenAI).
    
    Ưu tiên:
      1. Groq  → nhanh nhất, miễn phí tier rộng, dùng Llama3
      2. OpenAI → fallback nếu Groq không có key
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
        
        # Mở khóa SecretStr (của Pydantic V2) thành Chuỗi thật để dùng
        groq_key = settings.GROQ_API_KEY.get_secret_value() if getattr(settings, "GROQ_API_KEY", None) else os.getenv("GROQ_API_KEY", "")
        openai_key = settings.OPENAI_API_KEY.get_secret_value() if getattr(settings, "OPENAI_API_KEY", None) else os.getenv("OPENAI_API_KEY", "")

        if provider:
            self.provider = provider
        elif env_provider == "ollama":
            self.provider = "ollama"
        elif env_provider == "openai" or openai_key:
            self.provider = "openai"
        else:
            self.provider = "groq" # Mặc định

        # 2. Khởi tạo Client tương ứng
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=groq_key)
            self.model = model or getattr(settings, "LLM_CONTEXTUAL_MODEL", "llama-3.1-8b-instant")
            logger.info(f"LLMClient → Groq | model={self.model}")

        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=openai_key)
            self.model = model or getattr(settings, "LLM_MODEL_NAME", "gpt-4o-mini")
            logger.info(f"LLMClient → OpenAI | model={self.model}")

        elif self.provider == "ollama":
            from openai import OpenAI
            base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self._client = OpenAI(api_key="ollama", base_url=base_url)
            # Dùng Qwen 2.5 mà bạn đã có sẵn
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
        
        # Chọn model phù hợp: Dùng model lớn nếu là Groq, còn không thì dùng model mặc định của provider (Ollama)
        if self.provider == "groq":
            target_model = getattr(settings, "LLM_MODEL_NAME", "llama-3.3-70b-versatile")
        else:
            target_model = self.model
        
        try:
            stream = self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Lỗi Streaming LLM: {e}")
            yield f"Lỗi Streaming: {str(e)}"
