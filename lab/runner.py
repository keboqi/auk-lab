import asyncio
import copy
import time
from contextlib import suppress

from .audio import describe_audio, normalize_result
from .client import MODELS, RUNTIME, SGLANG_REVISION, BackendClient, build_request
from .schema import Review, RunRequest
from .store import Store, new_id, now

TERMINAL = {"completed", "failed", "partial", "cancelled", "interrupted"}


class Runner:
    def __init__(self, store: Store, client: BackendClient):
        self.store, self.client = store, client
        self.queue = asyncio.Queue(maxsize=20)
        self.jobs = {job["id"]: job for job in store.list("jobs")}
        for job in self.jobs.values():
            if job["status"] not in TERMINAL:
                job.update(status="interrupted", finished_at=now(), error="The app restarted before this job finished")
                store.write("jobs", job)
        self.worker = None

    def source(self, request: RunRequest):
        if not request.source_id:
            return None, None
        metadata = self.store.read("sources", request.source_id)
        return metadata, self.store.path("sources", request.source_id, ".wav").read_bytes()

    def prepare(self, request: RunRequest):
        metadata, audio = self.source(request)
        return build_request(request, request.models[0], audio, metadata["duration"] if metadata else None, request.seed)

    def submit(self, request: RunRequest) -> dict:
        self.prepare(request)
        if self.queue.full():
            raise ValueError("The queue is full (20 jobs). Wait for a job to finish")
        record = dict(id=new_id(), created_at=now(), status="queued", request=request.model_dump(), results=[],
                      cancel_requested=False, progress="Waiting for the evaluation worker")
        self.store.write("jobs", record)
        self.jobs[record["id"]] = record
        self.queue.put_nowait(record["id"])
        return record

    def get(self, identifier: str):
        return self.jobs[identifier]

    def save(self, job: dict):
        self.store.write("jobs", job)

    def cancel(self, identifier: str):
        job = self.get(identifier)
        if job["status"] in TERMINAL:
            return job
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job.update(status="cancelled", finished_at=now(), progress="Cancelled before inference")
        else:
            job.update(status="cancelling", progress="Waiting for the current backend call to finish; remaining runs will be skipped")
        self.save(job)
        return job

    def review(self, identifier: str, result_id: str, review: Review):
        job = self.get(identifier)
        result = next((item for item in job["results"] if item["id"] == result_id), None)
        if result is None:
            raise KeyError(result_id)
        result["review"] = review.model_dump()
        self.save(job)
        return result

    async def start(self):
        self.worker = asyncio.create_task(self.loop())

    async def close(self):
        if self.worker:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker
        await self.client.close()

    async def loop(self):
        while True:
            identifier = await self.queue.get()
            job = self.get(identifier)
            try:
                if not job["cancel_requested"]:
                    await self.run(job)
            except asyncio.CancelledError:
                job.update(status="interrupted", finished_at=now(), progress="App stopped; upstream inference may still be finishing")
                self.save(job)
                raise
            except Exception as exc:
                job.update(status="failed", finished_at=now(), error=str(exc), progress="Evaluation failed")
                self.save(job)
            finally:
                self.queue.task_done()

    async def run(self, job: dict):
        request = RunRequest(**job["request"])
        metadata, source = self.source(request)
        job.update(status="running", started_at=now(), progress="Checking model services")
        self.save(job)
        readiness = {model: await self.client.status(model) for model in request.models}
        if job["cancel_requested"]:
            job.update(status="cancelled", finished_at=now())
            self.save(job)
            return
        # Fail the whole comparison up front instead of producing a misleading one-sided A/B.
        offline = [model for model in request.models if not readiness[model]["ready"]]
        if offline:
            raise ValueError("Model service unavailable: " + "; ".join(f"{model}: {readiness[model]['error']}" for model in offline))
        for repeat in range(request.repeats):
            for model in request.models:
                if job["cancel_requested"]:
                    break
                seed = request.seed + repeat
                route, payload, duration = build_request(request, model, source, metadata["duration"] if metadata else None, seed)
                job["progress"] = f"{model.title()} · take {repeat + 1}/{request.repeats} · generating {duration:g}s"
                self.save(job)
                result = dict(id=new_id(), model=model, model_id=MODELS[model], seed=seed, take=repeat + 1,
                              created_at=now(), requested_duration=duration, route=route,
                              sglang_revision=SGLANG_REVISION, review=Review().model_dump())
                result["runtime"] = RUNTIME
                # Save exact inference controls without duplicating the base64 source in every manifest.
                recorded = copy.deepcopy(payload)
                if "ref_audio" in recorded:
                    recorded["ref_audio"] = {"source_id": request.source_id}
                if "ref_audio" in recorded.get("metadata", {}).get("tts_params", {}):
                    recorded["metadata"]["tts_params"]["ref_audio"] = {"source_id": request.source_id}
                result["payload"] = recorded
                started = time.perf_counter()
                try:
                    audio, backend_meta = await self.client.generate(model, route, payload)
                    elapsed = time.perf_counter() - started
                    # User cancellation doesn't claim to abort upstream GPU work or release the worker early.
                    if job["cancel_requested"]:
                        break
                    output = normalize_result(audio)
                    info = describe_audio(output)
                    self.store.path("outputs", result["id"], ".wav").write_bytes(output)
                    result.update(status="completed", audio_url=f"/media/outputs/{result['id']}.wav", **info,
                                  latency_seconds=round(elapsed, 4), rtf=round(elapsed / info["duration"], 4),
                                  duration_error_ms=round((info["duration"] - duration) * 1000, 2), backend_meta=backend_meta)
                except Exception as exc:
                    result.update(status="failed", error=str(exc), latency_seconds=round(time.perf_counter() - started, 4))
                job["results"].append(result)
                self.save(job)
        errors = sum(result["status"] == "failed" for result in job["results"])
        status = "failed" if errors == len(job["results"]) else "partial" if errors else "completed"
        job.update(status="cancelled" if job["cancel_requested"] else status, finished_at=now(), progress="Evaluation finished")
        self.save(job)
