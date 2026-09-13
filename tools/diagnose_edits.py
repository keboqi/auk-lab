"""Real editing probes via the app; reports acoustic evidence without grading semantics."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.client import build_request
from lab.schema import RunRequest

DEFAULT_INSTRUCTION = "把'跑车'替换为'嘉年华'"
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'interrupted'}


def compare_audio(source, output):
    x, xr = sf.read(io.BytesIO(source), always_2d=True)
    y, yr = sf.read(io.BytesIO(output), always_2d=True)
    if xr != yr or x.shape[1] != y.shape[1]:
        raise ValueError('Comparison requires matching sample rates and channels')
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Empty or nonfinite audio')
    n = min(len(x), len(y))
    rms_x, rms_y = np.sqrt(np.mean(x*x)), np.sqrt(np.mean(y*y))
    denominator = float(np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]))
    return dict(exact_samples_equal=x.shape == y.shape and bool(np.array_equal(x, y)),
                duration_delta_ms=round((len(y)-len(x))/xr*1000, 3),
                rms_delta_db=round(20*math.log10(rms_y/rms_x), 3) if min(rms_x, rms_y) > 1e-10 else None,
                unaligned_cosine=round(float(np.sum(x[:n]*y[:n]))/denominator, 6) if denominator > 1e-15 else None,
                relative_error=round(float(np.linalg.norm(x[:n]-y[:n])/np.linalg.norm(x[:n])), 6) if rms_x > 1e-10 else None)


def wait_job(client, identifier, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get('/api/jobs/' + identifier)
        response.raise_for_status()
        job = response.json()
        if job['status'] in TERMINAL:
            return job
        time.sleep(2)
    raise TimeoutError(f'Job {identifier} is still running; inspect it in the app. It has not been cancelled.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id', required=True)
    parser.add_argument('--model', choices=['flash', 'base'], default='flash')
    parser.add_argument('--backend', choices=['sglang', 'official'], default='sglang')
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
    parser.add_argument('--url', default='http://127.0.0.1:7865')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--emit-request', action='store_true')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url.rstrip('/'), timeout=30) as client:
        response = client.get('/api/sources')
        response.raise_for_status()
        metadata = next((s for s in response.json() if s['id'] == args.source_id), None)
        if metadata is None:
            raise SystemExit('Source ID not found. Use the source_id from a saved job.')
        response = client.get(metadata['audio_url'])
        response.raise_for_status()
        source = response.content
        request = RunRequest(task='custom', backend=args.backend, models=[args.model], source_id=args.source_id,
                             instruction=args.instruction, seed=args.seed)
        if args.emit_request:
            print(json.dumps(build_request(request, args.model, source, metadata['duration'], args.seed)[1], ensure_ascii=False))
            return
        output_dir = Path(args.output_dir or Path(os.environ.get('AUK_LAB_DATA', 'data')) / 'diagnostics' / uuid.uuid4().hex)
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / 'source.wav').write_bytes(source)
        report = dict(source_id=args.source_id, model=args.model, backend=args.backend, seed=args.seed, probes=[],
                      limitation='Waveform differences do not prove instruction following. Listen to the content edit and opposite volume edits. Cosine is unaligned and sensitive to time shifts.')
        outputs = {}
        # Opposite instructions with the same source/seed expose prompt-insensitive output.
        for label, instruction in [('content', args.instruction), ('quiet', '将音量降低10分贝。'), ('loud', '将音量提高10分贝。')]:
            request.instruction = instruction
            response = client.post('/api/jobs', json=request.model_dump())
            response.raise_for_status()
            identifier = response.json()['id']
            print(f'{label}: job {identifier}', flush=True)
            job = wait_job(client, identifier, args.timeout)
            (output_dir / f'{label}-job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
            probe = dict(label=label, job_id=identifier, instruction=instruction, status=job['status'])
            for result in job['results']:
                if result['status'] == 'completed':
                    response = client.get(result['audio_url'])
                    response.raise_for_status()
                    outputs[label] = response.content
                    (output_dir / f'{label}.wav').write_bytes(response.content)
                    probe['source_comparison'] = compare_audio(source, response.content)
                else:
                    probe['error'] = result.get('error')
            report['probes'].append(probe)
            (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if 'quiet' in outputs and 'loud' in outputs:
            report['quiet_vs_loud'] = compare_audio(outputs['quiet'], outputs['loud'])
        (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'Artifacts: {output_dir}; all probes also appear in the app.')
        if any(p['status'] != 'completed' for p in report['probes']):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
