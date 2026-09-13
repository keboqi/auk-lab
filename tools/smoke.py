"""Run a real short synthesis against the lab; never substitutes mock audio."""
import argparse
import json
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7865")
    parser.add_argument("--model", choices=["flash", "base", "both"], default="flash")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=30) as client:
        response = client.post("/api/jobs", json={"task": "voice_design", "models": ["flash", "base"] if args.model == "both" else [args.model],
                                                "text": "Welcome to the speech laboratory.",
                                                "instruction": "A clear, calm, natural speaking voice.", "duration": 3, "seed": 1234})
        response.raise_for_status()
        identifier = response.json()["id"]
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            response = client.get(f"/api/jobs/{identifier}")
            response.raise_for_status()
            job = response.json()
            if job["status"] in {"completed", "partial", "failed", "cancelled", "interrupted"}:
                print(json.dumps(job, ensure_ascii=False, indent=2))
                if job["status"] != "completed":
                    raise SystemExit(1)
                for result in job["results"]:
                    audio = client.get(result["audio_url"])
                    audio.raise_for_status()
                    if len(audio.content) < 100 or result["duration"] <= 0:
                        raise SystemExit("Missing or empty generated audio")
                return
            time.sleep(2)
    raise SystemExit("Smoke test timed out; inspect the job in the app")


if __name__ == "__main__":
    main()
