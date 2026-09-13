import asyncio
import copy
import csv
import hashlib
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .audio import prepare_upload
from .catalog import TASKS
from .client import SGLANG_REVISION, BackendClient
from .runner import Runner
from .schema import Review, RunRequest
from .store import Store, new_id, now

ROOT = Path(__file__).resolve().parent.parent


def create_app(data_dir=None, backend_client=None):
    @asynccontextmanager
    async def lifespan(app):
        store = Store(Path(data_dir or os.environ.get("AUK_LAB_DATA", ROOT / "data")))
        client = backend_client or BackendClient({
            "flash": os.environ.get("AUK_FLASH_URL", "http://127.0.0.1:8101"),
            "base": os.environ.get("AUK_BASE_URL", "http://127.0.0.1:8102"),
        })
        app.state.store = store
        app.state.runner = Runner(store, client)
        app.state.upload_slots = asyncio.Semaphore(2)
        await app.state.runner.start()
        yield
        await app.state.runner.close()

    app = FastAPI(title="AuK Lab", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/catalog")
    async def catalog():
        return dict(tasks=TASKS, max_seconds=30, sample_rate=24000, sglang_revision=SGLANG_REVISION,
                    hardware="RTX PRO 6000 Blackwell · 96 GB", sampling={"flash": "4 steps · CFG 0", "base": "32 steps · CFG 2"})

    @app.get("/api/status")
    async def status():
        results = await asyncio.gather(*(app.state.runner.client.status(model) for model in ("flash", "base")))
        return dict(models=dict(zip(("flash", "base"), results)), queued=app.state.runner.queue.qsize())

    @app.post("/api/sources")
    async def upload(file: UploadFile = File(...), start: float = Form(0), end: float | None = Form(None)):
        store = app.state.store
        identifier = new_id()
        raw = store.path("sources", identifier, ".upload")
        target = store.path("sources", identifier, ".wav")
        try:
            async with app.state.upload_slots:
                size = 0
                with raw.open("wb") as output:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > 50 * 1024 * 1024:
                            raise HTTPException(413, "Upload is limited to 50 MB")
                        output.write(chunk)
                info = await prepare_upload(raw, target, start, end)
                record = dict(id=identifier, name=(file.filename or "audio")[:200], created_at=now(),
                              start=start, end=end, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                              audio_url=f"/media/sources/{identifier}.wav", **info)
                store.write("sources", record)
                return record
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            raw.unlink(missing_ok=True)
            if not store.path("sources", identifier, ".json").exists():
                target.unlink(missing_ok=True)

    @app.get("/api/sources")
    async def sources():
        return app.state.store.list("sources")

    @app.post("/api/preview")
    async def preview(request: RunRequest):
        try:
            route, payload, duration = app.state.runner.prepare(request)
            payload = copy.deepcopy(payload)
            if "ref_audio" in payload:
                payload["ref_audio"] = "[uploaded audio, encoded as a data URL]"
            if "ref_audio" in payload.get("metadata", {}).get("tts_params", {}):
                payload["metadata"]["tts_params"]["ref_audio"] = "[uploaded audio, encoded as a data URL]"
            return dict(route=route, payload=payload, duration=duration)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/jobs", status_code=202)
    async def submit(request: RunRequest):
        try:
            return app.state.runner.submit(request)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/jobs")
    async def jobs():
        return sorted(app.state.runner.jobs.values(), key=lambda job: job["created_at"], reverse=True)

    @app.get("/api/jobs/{identifier}")
    async def job(identifier: str):
        try:
            return app.state.runner.get(identifier)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.post("/api/jobs/{identifier}/cancel")
    async def cancel(identifier: str):
        try:
            return app.state.runner.cancel(identifier)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.put("/api/jobs/{identifier}/results/{result_id}/review")
    async def review(identifier: str, result_id: str, request: Review):
        try:
            return app.state.runner.review(identifier, result_id, request)
        except KeyError as exc:
            raise HTTPException(404, "Result not found") from exc

    @app.post("/api/jobs/{identifier}/results/{result_id}/source")
    async def reuse_output(identifier: str, result_id: str):
        try:
            job = app.state.runner.get(identifier)
            result = next(item for item in job["results"] if item["id"] == result_id and item["status"] == "completed")
        except (KeyError, StopIteration) as exc:
            raise HTTPException(404, "Completed result not found") from exc
        store = app.state.store
        source_id = new_id()
        data = store.path("outputs", result_id, ".wav").read_bytes()
        store.path("sources", source_id, ".wav").write_bytes(data)
        record = dict(id=source_id, name=f"{job['request']['task']} · {result['model']} · seed {result['seed']}",
                      created_at=now(), sha256=hashlib.sha256(data).hexdigest(),
                      parent_job=identifier, parent_result=result_id,
                      audio_url=f"/media/sources/{source_id}.wav",
                      **{key: result[key] for key in ("duration", "sample_rate", "channels", "peak_dbfs", "rms_dbfs", "clipped_percent")})
        store.write("sources", record)
        return record

    @app.get("/media/{group}/{filename}")
    async def media(group: str, filename: str):
        if group not in {"sources", "outputs"} or not filename.endswith(".wav"):
            raise HTTPException(404)
        try:
            path = app.state.store.path(group, filename[:-4], ".wav")
        except ValueError as exc:
            raise HTTPException(404) from exc
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav", filename=filename, content_disposition_type="inline")

    @app.get("/api/export")
    async def export(format: str = "json"):
        records = await jobs()
        if format == "json":
            data = dict(exported_at=now(), sglang_revision=SGLANG_REVISION,
                        sources=app.state.store.list("sources"), jobs=records)
            return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json",
                            headers={"Content-Disposition": 'attachment; filename="auk-lab-results.json"'})
        if format != "csv":
            raise HTTPException(422, "Choose json or csv")
        output = io.StringIO(newline="")
        fields = ["job_id", "task", "model", "seed", "status", "latency_seconds", "duration", "rtf",
                  "duration_error_ms", "peak_dbfs", "rms_dbfs", "clipped_percent", "verdict", "content",
                  "identity", "instruction", "quality", "notes", "error"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            for result in record["results"]:
                row = dict(job_id=record["id"], task=record["request"]["task"], **result)
                row.update(result.get("review", {}))
                # Keep free-form review/error text from becoming spreadsheet formulas.
                row = {key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
                       for key, value in row.items()}
                writer.writerow(row)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="auk-lab-results.csv"'})

    return app


app = create_app()
