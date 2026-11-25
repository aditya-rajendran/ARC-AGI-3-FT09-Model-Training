"""Model connections for FT09. Heavy/optional packages load only on demand."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import base64
import gc
import io
import json
import time

from ft09_shared import build_turn_text, parse_answer


class _Backend:
    def __init__(self, model_name, max_new_tokens=512):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.log_path = None
        self.request_log = []
        self.context = {}

    def set_context(self, **context):
        self.context = context

    def _log(self, entry):
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name, **self.context, **entry,
        }
        self.request_log.append(entry)
        if self.log_path is not None:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _finish(self, reply, tokens, text, prompt, memory_note, temperature, started, truncated=False):
        move = parse_answer(reply)
        self._log({
            "status": "ok" if move is not None else "invalid_reply",
            "user_text": text, "system_prompt": prompt,
            "memory_in": memory_note, "reply": reply,
            "temperature": temperature, "completion_tokens": tokens,
            "truncated": truncated, "elapsed_seconds": time.monotonic() - started,
        })
        if move is None:
            print("Invalid model reply:", repr(reply), flush=True)
        if truncated:
            print("The reply reached the output-token limit; check for truncation.", flush=True)
        return move, int(tokens or 0)


class OpenAIBackend(_Backend):
    """An OpenAI-compatible university endpoint, including LISA Pro."""
    def __init__(self, model_name, api_key, base_url, *, timeout=180.0, max_new_tokens=512):
        from openai import OpenAI
        super().__init__(model_name, max_new_tokens)
        # Retries belong to data_pipeline.run_attempt, not an invisible second layer.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        self.base_url = base_url
        self.timeout = timeout

    def info(self):
        return {
            "model_id": self.model_name, "connection": "openai-compatible",
            "base_url": self.base_url, "timeout_seconds": self.timeout,
            "max_new_tokens": self.max_new_tokens,
        }

    def ask(self, image, text_grid, history, prompt, memory_note, temperature, format_reminder):
        from openai import APITimeoutError
        text = build_turn_text(text_grid, history, memory_note, format_reminder)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ]},
                ],
                temperature=temperature, max_tokens=self.max_new_tokens,
            )
        except Exception as exc:
            timed_out = isinstance(exc, APITimeoutError)
            self._log({
                "status": "timeout" if timed_out else "error",
                "user_text": text, "system_prompt": prompt, "memory_in": memory_note,
                "temperature": temperature, "completion_tokens": None,
                # Avoid recording exception text that may contain request headers.
                "error_type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - started,
            })
            if timed_out:
                raise TimeoutError("Model request timed out.") from exc
            raise
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "completion_tokens", None)
        choice = response.choices[0]
        reply = choice.message.content or ""
        return self._finish(
            reply, tokens, text, prompt, memory_note, temperature, started,
            truncated=getattr(choice, "finish_reason", None) == "length",
        )

    def close(self):
        self.client.close()


class LocalQwenBackend(_Backend):
    """Qwen3.5 on one CUDA GPU, with the same interface as the API backend."""
    def __init__(self, model_name="Qwen/Qwen3.5-2B", *, max_new_tokens=512):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        super().__init__(model_name, max_new_tokens)
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. Start JupyterHub with a GPU.")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.gpu = torch.cuda.get_device_name(0)
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        print(f"Loading {model_name} on {self.gpu} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name, dtype=self.dtype, device_map={"": 0}, attn_implementation="sdpa",
        )
        self.model.eval()

    def info(self):
        return {
            "model_id": self.model_name, "connection": "local-transformers",
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "dtype": str(self.dtype), "gpu": self.gpu,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "thinking": False, "max_new_tokens": self.max_new_tokens,
            "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0,
        }

    def ask(self, image, text_grid, history, prompt, memory_note, temperature, format_reminder):
        import torch
        text = build_turn_text(text_grid, history, memory_note, format_reminder)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": text},
            ]},
        ]
        started = time.monotonic()
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=temperature,
                    top_p=0.8, top_k=20, repetition_penalty=1.0,
                )
            new_tokens = output[0, input_length:]
            reply = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:
            self._log({
                "status": "error", "user_text": text, "system_prompt": prompt,
                "memory_in": memory_note, "temperature": temperature,
                "completion_tokens": None, "error_type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - started,
            })
            raise
        return self._finish(
            reply, len(new_tokens), text, prompt, memory_note, temperature, started,
            truncated=len(new_tokens) >= self.max_new_tokens,
        )

    def close(self):
        import torch
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
