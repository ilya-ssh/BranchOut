import os
import httpx
import json
import time
import hashlib
from collections import OrderedDict
from contextvars import ContextVar
from datetime import datetime
from typing import Dict, Optional, Any, List, Union, Callable, Tuple

from src.utils.decorators import infinite_retry_with_backoff
from src.logger import api_logger


TRACE_HOOK: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar("TRACE_HOOK", default=None)
def _strip_response_format_for_model(model: str) -> bool:
    csv = os.getenv("STRIP_RESPONSE_FORMAT_MODELS", "z-ai/glm-5")
    needles = [x.strip() for x in csv.split(",") if x.strip()]
    m = (model or "").lower()
    return any(n.lower() in m for n in needles)

def _maybe_trace(event: Dict[str, Any]) -> None:
    hook = TRACE_HOOK.get()
    if hook is None:
        return
    try:
        hook(event)
    except Exception:
        pass


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


def _extract_usage_total_tokens(obj: Dict[str, Any]) -> int:
    usage = obj.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if total is None:
        total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    try:
        return int(total or 0)
    except Exception:
        return 0


def _pick_last_user(messages: List[Dict[str, Any]]) -> str:
    last_user = ""
    for m in reversed(messages or []):
        if m.get("role") == "user" and m.get("content") is not None:
            last_user = _safe_str(m.get("content"))
            if last_user:
                break
    return last_user


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        #опц хэдэры опенрутер
        referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE") or os.getenv("X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        if extra_headers:
            headers.update(extra_headers)

        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            trust_env=False,
        )


        self.call_count = 0
        self.completion_calls = 0
        self.embedding_calls = 0
        self.image_calls = 0

        self.total_tokens_used = 0

        #чек тулы
        self.supports_functions = str(os.getenv("LLM_SUPPORTS_FUNCTIONS", "true")).lower() in ("1", "true", "yes", "on")

        try:
            self._embed_cache_max = int(os.getenv("EMBED_CACHE_MAX", "2048"))
        except Exception:
            self._embed_cache_max = 2048
        self._embed_cache: "OrderedDict[str, List[float]]" = OrderedDict()

    def _next_call_id(self, operation_name: str) -> str:
        self.call_count += 1
        ts = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        return f"{operation_name}_{self.call_count}_{ts}"

    def _embed_cache_get(self, key: str) -> Optional[List[float]]:
        if not key:
            return None
        v = self._embed_cache.get(key)
        if v is None:
            return None
        self._embed_cache.move_to_end(key, last=True)
        return v

    def _embed_cache_put(self, key: str, vec: List[float]) -> None:
        if not key:
            return
        self._embed_cache[key] = vec
        self._embed_cache.move_to_end(key, last=True)
        while len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

    @staticmethod
    def _store_full_prompts_enabled() -> bool:
        return str(os.getenv("TRACE_STORE_FULL_PROMPTS", "false")).lower() in ("1", "true", "yes", "on")

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_completion(
        self,
        *,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 40000,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        operation_name: str = "unknown",
        response_format: Optional[Dict[str, str]] = None,
        supports_tools: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.completion_calls += 1
        call_id = self._next_call_id(operation_name)
        t0 = time.perf_counter()
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt is not None:
                messages.append({"role": "user", "content": prompt})
        api_logger.info(f"=== API CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Model: {model}")
        store_full = self._store_full_prompts_enabled()
        last_user = _pick_last_user(messages)
        _maybe_trace(
            {
                "type": "llm.request",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages_count": len(messages),
                "last_user_preview": last_user[:300],
                "last_user_sha256": _sha(last_user),
                "messages": messages if store_full else None,
            }
        )
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        use_tools = self.supports_functions if supports_tools is None else bool(supports_tools)
        if use_tools:
            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        strip_rf = bool(response_format) and _strip_response_format_for_model(model)
        if strip_rf:
            json_instr = "Return ONLY one valid JSON object. No markdown. No extra text."
            messages2 = [dict(m) for m in (messages or [])]
            if messages2 and messages2[0].get("role") == "system" and isinstance(messages2[0].get("content"), str):
                messages2[0]["content"] = messages2[0]["content"].rstrip() + "\n\n" + json_instr
            else:
                messages2.insert(0, {"role": "system", "content": json_instr})
            payload["messages"] = messages2
        else:
            if response_format:
                payload["response_format"] = response_format

        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body_preview = ""
            try:
                body_preview = (response.text or "")[:2000]
            except Exception:
                body_preview = ""
            api_logger.error(f"HTTP error in {operation_name} ({call_id}): {e}. Body preview: {body_preview}")
            raise

        result = response.json()

        if "choices" not in result or not result["choices"]:
            api_logger.error(f"API returned response without choices array: {result}")
            raise ValueError("API returned response without choices array")

        choice0 = result["choices"][0] or {}
        message = choice0.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        content = _safe_str(message.get("content"))
        message["content"] = content

        if response_format and str(response_format.get("type") or "").lower() == "json_object":
            if not (content or "").strip() and not tool_calls:
                finish_reason = choice0.get("finish_reason")
                api_logger.error(
                    f"Empty content in json_object mode. call_id={call_id} "
                    f"finish_reason={finish_reason} choice0_keys={list(choice0.keys())} message_keys={list(message.keys())}"
                )
                raise ValueError("Empty content returned in json_object mode")

        tokens_used = _extract_usage_total_tokens(result)
        self.total_tokens_used += int(tokens_used or 0)

        dt = time.perf_counter() - t0
        api_logger.info(f"Response: {len(content)} chars, {tokens_used} tokens, tool_calls={len(tool_calls)}")
        api_logger.info(f"Cumulative tokens: {self.total_tokens_used}")
        api_logger.info(f"=== API CALL {call_id} END ===\n")

        _maybe_trace(
            {
                "type": "llm.response",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": model,
                "latency_s": dt,
                "tokens_used": int(tokens_used or 0),
                "tool_calls_count": len(tool_calls),
                "content_preview": (content or "")[:500],
                "content_sha256": _sha(content or ""),
                "raw": result if store_full else None,
            }
        )

        return result

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        operation_name: str = "embedding",
    ) -> List[float]:
        self.embedding_calls += 1
        embed_model = model or os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
        key = f"{embed_model}:{_sha(_safe_str(text))}"

        cached = self._embed_cache_get(key)
        if cached is not None:
            _maybe_trace(
                {
                    "type": "embed.cache_hit",
                    "operation_name": operation_name,
                    "model": embed_model,
                    "text_sha256": _sha(_safe_str(text)),
                    "len": len(cached),
                }
            )
            return cached

        call_id = self._next_call_id(operation_name)
        t0 = time.perf_counter()

        api_logger.info(f"=== EMBEDDING CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Embed model: {embed_model}")

        store_full = self._store_full_prompts_enabled()
        text_s = _safe_str(text)

        _maybe_trace(
            {
                "type": "embed.request",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": embed_model,
                "text_preview": (text_s or "")[:300],
                "text_sha256": _sha(text_s),
                "text": text_s if store_full else None,
            }
        )

        payload = {"model": embed_model, "input": text_s}

        response = await self.client.post(f"{self.base_url}/embeddings", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body_preview = ""
            try:
                body_preview = (response.text or "")[:2000]
            except Exception:
                body_preview = ""
            api_logger.error(f"HTTP error in {operation_name} ({call_id}): {e}. Body preview: {body_preview}")
            raise

        data = response.json()

        if "data" not in data or not data["data"]:
            api_logger.error(f"Embedding API returned no data: {data}")
            raise ValueError("Embedding API returned no data")

        embedding = (data["data"][0] or {}).get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Embedding format unexpected")

        tokens_used = _extract_usage_total_tokens(data)
        if tokens_used:
            self.total_tokens_used += tokens_used

        api_logger.info(f"Embedding length: {len(embedding)}")
        api_logger.info(f"=== EMBEDDING CALL {call_id} END ===\n")

        dt = time.perf_counter() - t0
        _maybe_trace(
            {
                "type": "embed.response",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": embed_model,
                "latency_s": dt,
                "len": len(embedding),
                "tokens_used": int(tokens_used or 0),
            }
        )

        self._embed_cache_put(key, embedding)
        return embedding

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        aspect_ratio: str = "1:1",
        operation_name: str = "image_generation",
    ) -> Dict[str, Any]:
        self.image_calls += 1
        image_model = model or os.getenv("IMAGE_MODEL_NAME", "google/gemini-2.5-flash-image")
        call_id = self._next_call_id(operation_name)
        t0 = time.perf_counter()

        api_logger.info(f"=== IMAGE CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Image model: {image_model}")

        store_full = self._store_full_prompts_enabled()
        prompt_s = _safe_str(prompt)

        _maybe_trace(
            {
                "type": "image.request",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": image_model,
                "aspect_ratio": aspect_ratio,
                "prompt_preview": (prompt_s or "")[:300],
                "prompt_sha256": _sha(prompt_s or ""),
                "prompt": prompt_s if store_full else None,
            }
        )

        payload = {
            "model": image_model,
            "messages": [{"role": "user", "content": prompt_s}],
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": aspect_ratio},
            "stream": False,
        }

        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body_preview = ""
            try:
                body_preview = (response.text or "")[:2000]
            except Exception:
                body_preview = ""
            api_logger.error(f"HTTP error in {operation_name} ({call_id}): {e}. Body preview: {body_preview}")
            raise

        data = response.json()

        if "choices" not in data or not data["choices"]:
            api_logger.error(f"Image API returned no choices: {data}")
            raise ValueError("Image API returned no choices")

        message = (data["choices"][0] or {}).get("message") or {}
        images = message.get("images") or []
        if not isinstance(images, list) or not images:
            api_logger.error(f"Image API returned no images: {data}")
            raise ValueError("Image API returned no images")

        img_obj = images[0]
        if not isinstance(img_obj, dict):
            img_obj = {"raw_image_obj": img_obj}

        tokens_used = _extract_usage_total_tokens(data)
        if tokens_used:
            self.total_tokens_used += tokens_used

        api_logger.info(f"Image object keys: {list(img_obj.keys())}")
        api_logger.info(f"=== IMAGE CALL {call_id} END ===\n")

        dt = time.perf_counter() - t0
        _maybe_trace(
            {
                "type": "image.response",
                "call_id": call_id,
                "operation_name": operation_name,
                "model": image_model,
                "latency_s": dt,
                "tokens_used": int(tokens_used or 0),
                "has_image_url": bool((img_obj.get("image_url") or {}).get("url")) if isinstance(img_obj, dict) else False,
            }
        )

        return img_obj

    async def close(self) -> None:
        await self.client.aclose()