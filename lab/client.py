import base64
import binascii
import math
import json
from pathlib import Path

import httpx

from .catalog import BY_ID
from .schema import RunRequest

RUNTIME = json.loads(Path(__file__).with_name("runtime.json").read_text(encoding="utf-8"))
MODELS = {key: value["id"] for key, value in RUNTIME["models"].items()}
SGLANG_REVISION = RUNTIME["sglang_revision"]


def build_request(request: RunRequest, model: str, source: bytes | None, source_seconds: float | None,
                  seed: int) -> tuple[str, dict, float]:
    duration = request.duration
    if duration is None:
        if source_seconds is None:
            raise ValueError("An output duration or source clip is required")
        duration = source_seconds / request.speed_factor if request.task == "speed" else source_seconds
    duration = math.ceil(round(duration * 50, 8)) / 50
    if not 0.02 <= duration <= 30:
        raise ValueError(f"Output would be {duration:g}s. Adjust duration/speed or use a shorter source (maximum 30s)")
    audio = "data:audio/wav;base64," + base64.b64encode(source).decode() if source else None
    stage_params = {"auk_engine": {"gen_seconds": duration}}
    if BY_ID[request.task]["route"] == "speech":
        payload = dict(model=MODELS[model], input=request.text.strip(), response_format="wav",
                       stream=False, seed=seed, stage_params=stage_params)
        if audio:
            payload["ref_audio"] = audio
        else:
            payload["instructions"] = request.instruction.strip()
        return "/v1/audio/speech", payload, duration
    # The raw endpoint reads AuK's seed from metadata.tts_params, not a top-level seed.
    params = {"seed": seed}
    if audio:
        params["ref_audio"] = audio
    payload = dict(prompt=request.instruction.strip(), metadata={"tts_params": params},
                   stage_params=stage_params, output_modalities=["audio"], return_logprob=False, stream=False)
    return "/generate", payload, duration


def decode_edit_response(body: dict) -> tuple[bytes, dict]:
    audio = body.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
        raise ValueError("Expected /generate response audio.data with inline base64 audio; path-only output is unsupported")
    fmt = audio.get("format")
    if fmt not in {None, "wav", "flac", "mp3", "ogg", "opus"}:
        raise ValueError(f"Unsupported audio encoding: {fmt}")
    encoded = audio["data"]
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Backend returned invalid base64 audio") from exc
    meta = body.get("meta_info") or {}
    # Don't persist echoed input audio from request_metadata.
    return data, {key: meta[key] for key in ("finish_reason", "prompt_tokens", "completion_tokens", "weight_version") if key in meta}


class BackendClient:
    def __init__(self, urls: dict[str, str], *, transport=None):
        self.urls = {key: url.rstrip("/") for key, url in urls.items()}
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10), transport=transport)

    async def status(self, model: str) -> dict:
        try:
            response = await self.http.get(self.urls[model] + "/health", timeout=3)
            response.raise_for_status()
            models = await self.http.get(self.urls[model] + "/v1/models", timeout=3)
            models.raise_for_status()
            ids = [entry.get("id", "") for entry in models.json().get("data", [])]
            ready = MODELS[model] in ids
            return dict(ready=ready, models=ids, error=None if ready else "Endpoint is not serving " + MODELS[model])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            return dict(ready=False, error=str(exc) or type(exc).__name__)

    async def generate(self, model: str, route: str, payload: dict) -> tuple[bytes, dict]:
        response = await self.http.post(self.urls[model] + route, json=payload)
        if response.is_error:
            raise ValueError(f"{model} backend HTTP {response.status_code}: {response.text[:700]}")
        if route == "/generate":
            return decode_edit_response(response.json())
        return response.content, {}

    async def close(self):
        await self.http.aclose()
