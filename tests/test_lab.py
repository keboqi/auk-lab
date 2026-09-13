import asyncio
import base64
import io
import json
import time

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from lab.app import create_app
from lab.audio import describe_audio, normalize_result
from lab.catalog import TASKS
from lab.client import BackendClient, build_request, decode_edit_response
from lab.runner import Runner
from lab.schema import RunRequest
from lab.store import Store, new_id, now


def wav(seconds=0.2, rate=24000):
    output = io.BytesIO()
    samples = np.sin(np.arange(round(seconds * rate)) * 2 * np.pi * 440 / rate) * 0.1
    sf.write(output, samples, rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


def mock_handler(calls):
    def handle(request):
        model = "base" if request.url.host == "base" else "flash"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "tencent/AuK" if model == "base" else "tencent/AuK-Flash"}]})
        body = json.loads(request.content)
        calls.append((model, request.url.path, body))
        audio = wav(body["stage_params"]["auk_engine"]["gen_seconds"])
        if request.url.path == "/generate":
            return httpx.Response(200, json={"audio": {"data": base64.b64encode(audio).decode(), "format": "wav"},
                                           "meta_info": {"finish_reason": {"type": "stop"}, "request_metadata": body["metadata"]}})
        return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})
    return handle


def make_client(calls):
    return BackendClient({"flash": "http://flash", "base": "http://base"}, transport=httpx.MockTransport(mock_handler(calls)))


def wait_job(client, identifier):
    for _ in range(150):
        job = client.get(f"/api/jobs/{identifier}").json()
        if job["status"] in {"completed", "failed", "partial", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError("Job did not finish")


@pytest.mark.parametrize("task", TASKS, ids=[task["id"] for task in TASKS])
def test_every_catalog_task_maps_to_supported_endpoint(task):
    source_id = "a" * 32 if task["audio"] else None
    request = RunRequest(task=task["id"], source_id=source_id, text=task.get("text", ""),
                         instruction=task["instruction"], duration=task["duration"])
    route, payload, duration = build_request(request, "flash", wav() if source_id else None, 0.2 if source_id else None, 42)
    assert 0 < duration <= 30
    assert route == ("/v1/audio/speech" if task["route"] == "speech" else "/generate")
    assert "nfe" not in payload["stage_params"]["auk_engine"]
    if route == "/generate":
        assert payload["return_logprob"] is False
        assert payload["metadata"]["tts_params"]["seed"] == 42
        assert "seed" not in payload
    if task["id"] == "voice_clone":
        assert "instructions" not in payload and "ref_audio" in payload


def test_duration_validation_and_quantization():
    request = RunRequest(task="speed", source_id="a" * 32, instruction="Slow down", speed_factor=0.5)
    with pytest.raises(ValueError, match="maximum 30s"):
        build_request(request, "base", wav(), 20, 1)
    request.duration = 0.221
    assert build_request(request, "base", wav(), 20, 1)[2] == 0.24
    for duration in [0, -1, 31, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            RunRequest(task="voice_design", text="Hi", duration=duration)
    with pytest.raises(ValueError, match="explicit"):
        RunRequest(task="voice_design", text="Hi")


def test_audio_contract_rejects_path_only_and_malformed_audio():
    with pytest.raises(ValueError, match="path-only"):
        decode_edit_response({"audio": {"path": "/tmp/audio.wav"}})
    with pytest.raises(ValueError, match="base64"):
        decode_edit_response({"audio": {"data": "!!"}})
    with pytest.raises(ValueError):
        normalize_result(b"not audio")
    with pytest.raises(ValueError, match="24 kHz"):
        normalize_result(wav(rate=16000))


def test_upload_compare_review_export_and_restart(tmp_path):
    calls = []
    with TestClient(create_app(tmp_path, make_client(calls))) as client:
        assert client.get("/").status_code == 200
        uploaded = client.post("/api/sources", files={"file": ("reference.wav", wav(), "audio/wav")})
        assert uploaded.status_code == 200, uploaded.text
        source = uploaded.json()
        assert source["sample_rate"] == 24000
        request = dict(task="emotion", models=["flash", "base"], source_id=source["id"],
                       instruction="Make this happy", repeats=2, seed=5)
        preview = client.post("/api/preview", json=request)
        assert preview.status_code == 200
        assert "data:audio" not in preview.text
        response = client.post("/api/jobs", json=request)
        assert response.status_code == 202
        job = wait_job(client, response.json()["id"])
        assert job["status"] == "completed", job
        assert [(result["model"], result["seed"]) for result in job["results"]] == [("flash", 5), ("base", 5), ("flash", 6), ("base", 6)]
        assert all(result["duration_error_ms"] == 0 for result in job["results"])
        assert all("request_metadata" not in result["backend_meta"] for result in job["results"])
        result = job["results"][0]
        assert client.get(result["audio_url"]).status_code == 200
        reused = client.post(f"/api/jobs/{job['id']}/results/{result['id']}/source").json()
        assert reused["parent_result"] == result["id"]
        assert describe_audio(client.get(reused["audio_url"]).content)["duration"] == result["duration"]
        assert client.put(f"/api/jobs/{job['id']}/results/{result['id']}/review", json={"verdict": "keep", "quality": 4, "notes": "=unsafe-formula"}).status_code == 200
        export = client.get("/api/export").json()
        assert export["jobs"][0]["results"][0]["review"]["verdict"] == "keep"
        assert "data:audio" not in json.dumps(export)
        assert "'=unsafe-formula" in client.get("/api/export?format=csv").text
    with TestClient(create_app(tmp_path, make_client([]))) as client:
        recovered = client.get(f"/api/jobs/{job['id']}").json()
        assert recovered["status"] == "completed"
        assert recovered["results"][0]["review"]["quality"] == 4


def test_reject_long_source_bad_crop_missing_source_and_traversal(tmp_path):
    with TestClient(create_app(tmp_path, make_client([]))) as client:
        response = client.post("/api/sources", files={"file": ("long.wav", wav(31), "audio/wav")})
        assert response.status_code == 422 and "30 seconds" in response.text
        assert not list((tmp_path / "sources").iterdir())
        response = client.post("/api/sources", files={"file": ("long.wav", wav(31), "audio/wav")}, data={"start": "5", "end": "5.2"})
        assert response.status_code == 200
        assert response.json()["duration"] == pytest.approx(0.2)
        response = client.post("/api/sources", files={"file": ("bad.wav", wav(), "audio/wav")}, data={"start": "5", "end": "4"})
        assert response.status_code == 422
        assert client.post("/api/jobs", json={"task": "emotion", "instruction": "Happy", "source_id": "f" * 32}).status_code == 422
        assert client.post("/api/jobs", json={"task": "emotion", "instruction": "Happy", "source_id": "../.env"}).status_code == 422
        assert client.get("/media/outputs/not-an-id.wav").status_code == 404


def test_offline_comparison_fails_before_any_generation(tmp_path):
    calls = []
    async def offline(model):
        return {"ready": model == "flash", "error": "offline"}
    backend = make_client(calls)
    backend.status = offline
    with TestClient(create_app(tmp_path, backend)) as client:
        job = client.post("/api/jobs", json=dict(task="voice_design", models=["flash", "base"], text="Hi", duration=0.2)).json()
        assert wait_job(client, job["id"])["status"] == "failed"
        assert not calls


def test_cancel_waits_for_current_call_and_skips_other_takes(tmp_path):
    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []
        backend = make_client(calls)
        async def generate(model, route, payload):
            calls.append(model)
            started.set()
            await finish.wait()
            return wav(), {}
        backend.generate = generate
        runner = Runner(Store(tmp_path), backend)
        await runner.start()
        job = runner.submit(RunRequest(task="voice_design", text="Hi", duration=0.2, repeats=3))
        await asyncio.wait_for(started.wait(), 3)
        runner.cancel(job["id"])
        assert job["status"] == "cancelling"
        assert not runner.worker.done()
        finish.set()
        await asyncio.wait_for(runner.queue.join(), 3)
        assert job["status"] == "cancelled" and not job["results"]
        assert calls == ["flash"]
        await runner.close()
    asyncio.run(scenario())


def test_restart_marks_unfinished_job_interrupted(tmp_path):
    async def scenario():
        store = Store(tmp_path)
        identifier = new_id()
        store.write("jobs", dict(id=identifier, status="running", created_at=now()))
        backend = make_client([])
        runner = Runner(store, backend)
        assert runner.jobs[identifier]["status"] == "interrupted"
        assert store.read("jobs", identifier)["status"] == "interrupted"
        await runner.close()
    asyncio.run(scenario())
