# AuK Lab

A standalone listening and evaluation app for **AuK and AuK-Flash**, served by
**SGLang-Omni or the official Tencent AuK Python implementation** on a Linux
machine with an **RTX PRO 6000 Blackwell (96 GB)**.
It has no dependency on the IndexTTS app, its Python environment, ports, or data.

## Start on the GPU host

Requirements: a working NVIDIA driver, Docker Engine with Compose v2, NVIDIA
Container Toolkit, and enough free disk for the large CUDA image and model cache.
The host driver must support the CUDA version in the pinned image; startup prints
the driver, CUDA/PyTorch versions, device capacity, and a CUDA kernel check.

Clone the app on the Linux host, then run the bootstrap script:

```bash
git clone https://github.com/keboqi/auk-lab.git
cd auk-lab
bash quickstart.sh
```

The script checks Docker/NVIDIA access, creates `.env` if missing, builds the
selected runtime and UI, starts Flash and the UI, and waits for readiness.
Python, Conda, and CUDA packages do not need to be installed on the host.
Existing `.env` settings are preserved. Host drivers and Docker remain prerequisites.

```bash
bash quickstart.sh --model both                # Base + Flash comparisons
bash quickstart.sh --model base --gpu 0        # Base only
bash quickstart.sh --port 7870 --no-wait        # return after container startup
bash auk-lab.sh logs
```

Open **http://127.0.0.1:7865** on that machine. For a remote machine, forward the UI:

```bash
ssh -L 7865:127.0.0.1:7865 user@gpu-host
```

The UI starts independently of model loading and shows each model's readiness.
The first model start downloads pinned AuK and Qwen2.5-Omni-3B checkpoints into a
shared Docker volume. Subsequent starts reuse those snapshots.

Switch to Base, or start both models for A/B evaluation:

```bash
bash auk-lab.sh switch base           # stop Flash, start Base
bash auk-lab.sh up both               # both available; app evaluates sequentially
bash auk-lab.sh switch flash          # stop Base, start Flash
bash auk-lab.sh status
bash auk-lab.sh doctor
bash auk-lab.sh down                  # stops this project; keeps all data/cache
```

Both-model operation is an option for the 96 GB target, **not a measured VRAM
guarantee**. Start with short clips. If memory is insufficient, switch to a single
model, save the experiment, switch models, and use **Load settings** to repeat it.
Other GPU applications remain the host operator's responsibility; the launcher
does not stop them. No privileged container or host networking is required.

Defaults: UI `7865`, Flash `8101`, Base `8102`. All published ports bind to
loopback. The evaluation app has no authentication; use an SSH tunnel or trusted
LAN access rather than exposing it publicly.

## Compare the official Python backend with SGLang-Omni

Update an existing checkout and start one model variant on both engines:

```bash
git pull --ff-only
bash quickstart.sh --backend both --model flash
# Or compare the Base model:
bash quickstart.sh --backend both --model base
```

Select **Compare SGLang + Official** in the UI's **Inference backend** control,
then choose the matching model variant. Each take runs sequentially on both
engines with the same source, instruction, duration, and seed. Result cards,
history, inference records, WAV filenames, and exports identify the engine.
Existing experiments default to SGLang when their saved backend is absent.

For the smallest GPU memory footprint, use one engine at a time:

```bash
bash quickstart.sh --backend official --model flash
# After testing, switch to SGLang with the same saved clip:
bash quickstart.sh --backend sglang --model flash
```

Use **Load settings** on a saved experiment, change **Inference backend**, and
rerun. Switching stops unselected model services and preserves data/checkpoints.
Two resident engines' peak memory on the 96 GB card still needs measurement;
the bootstrap intentionally does not start all four model/engine combinations.
Official services publish loopback ports `8201` (Flash) and `8202` (Base).

The official adapter calls the unmodified `AukInfer.generate()` from Tencent's
source at `0dfd4d39015078351217b09a40e263355aaa646b`. It uses the same pinned AuK,
Flash, and Qwen checkpoints as SGLang. The image reuses the verified CUDA 13
platform image for Blackwell and preserves its Torch/Transformers versions,
instead of installing upstream's default Torch 2.7 dependency set. Thus this
compares the two model implementations on our CUDA stack, not every dependency
of Tencent's recommended environment. The official container runs no SGLang
model or scheduler. Its HTTP adapter implements only the lab's WAV, nonstreaming
requests; it is not a general OpenAI API server.

Both use raw instructions without the optional Prompt Enhancer, ASR, VAD, or
task-specific loudness processing. This keeps the first editing comparison
focused on model implementation. It does not reproduce the official Gradio
demo's optional preprocessing. Matching seeds do not imply bit-identical output
across implementations.

## Investigate edits that reproduce the source

The original launcher forced BF16 DiT weight storage. The default now follows
SGLang's upstream parity recipe: **FP32 DiT storage with BF16 autocast**. This
removes one numerical difference; it is **not a confirmed fix** for ignored edits.
`AUK_WEIGHT_DTYPE=bfloat16` in `.env` opts SGLang back into the former speed mode.
The official implementation always follows its FP32/BF16 recipe. Base sampling
can be set with `AUK_BASE_NFE` and `AUK_BASE_CFG`; Flash locks its released 4-step,
CFG-off recipe in both implementations. Restart services after changing settings.

After the selected backend is ready, run the following with a source ID from an
existing experiment:

```bash
bash diagnose-edits.sh YOUR_SOURCE_ID flash sglang
bash diagnose-edits.sh YOUR_SOURCE_ID flash official
```

The script runs `把'跑车'替换为'嘉年华'` plus opposite ±10 dB volume instructions
with seed 1234. Use that content probe on a recording containing 跑车, or customize:

```bash
docker compose exec -T lab python tools/diagnose_edits.py \
  --source-id YOUR_SOURCE_ID --model base --backend official \
  --instruction 'Replace the word actually present in your recording.'
```

Probes appear in the UI and save WAVs/job records under `data/diagnostics/`.
Reports measure sample equality, RMS changes, and unaligned cosine similarity.
These measurements do not grade instruction following: listen to the actual
words and contrastive edits. For SGLang, the shell script also checks its installed
request parser and Qwen processor on CPU, logging the instruction, template,
audio feature shapes, and token count without logging the inline audio bytes.
That inspection does not observe live worker hidden states. Nothing is uploaded
outside your configured inference services.

## What you can test

The 21 presets cover every advertised task family:

| Family | Presets |
| --- | --- |
| Generation | Voice design; reference-based voice cloning |
| Content | Spoken-word replacement/insertion/deletion; lyric editing |
| Acoustic | Pitch; speed; volume |
| Delivery | Emotion; timbre; accent reduction; nonverbal events; normal ↔ whisper |
| Restoration | Denoising; dereverberation; combined cleanup; bandwidth/recording repair |
| Separation | Speaker by order; speaker by quoted words; singing/human-voice extraction |
| Exploration | Arbitrary instruction, with optional source audio |

Every instruction is editable. For content edits, replace the example words with
words actually in your recording. Use dedicated noisy, overlapping, musical,
whispered, and accented clips to judge the respective tasks. A single clean TTS
reference is not a useful test of all transformations.

1. Choose a task and upload a clip, optionally setting crop start/end first.
2. Adjust the instruction and target duration. Inspect the request if needed.
3. Select the inference backend and Flash, Base, or both. Choose 1–5 takes;
   seeds increment per take and match between models/backends.
4. Listen to the experiment's source and outputs. Save content/identity/
   instruction/quality scores, a verdict, and notes.
5. Use an output as the next source to explore chained edits. Parent run IDs are
   retained. **Load settings** restores a saved experiment for another run.
6. Download WAVs and export JSON/CSV to decide what belongs in the main app.

## Inference behavior

- **Real backend only.** The normal application never generates placeholder audio.
- Generation uses `/v1/audio/speech`; editing uses `/generate`, with inline source
  audio, `output_modalities=["audio"]`, and `return_logprob=false`.
- Voice design and cloning require an explicit target duration. This avoids an
  extra transcription/ASR service. Source audio selects the cloned voice; use
  free-form mode for combined cloning and delivery instructions.
- For edits, blank duration matches the source. Speed uses source duration divided
  by the speed factor. An explicit duration overrides these rules. Requested
  durations round up to a 20 ms frame; the app rejects values beyond 30 seconds.
- Source clips also have a 30-second application limit. This is **not** ComfyUI's
  source-plus-target limit. Long files require an explicit crop and are never
  silently shortened. Upload limit: 50 MB.
- FFmpeg decodes/crops/resamples source audio to mono 24 kHz. It does not normalize
  loudness or trim silence. This intentionally makes acoustic comparisons visible.
- Outputs are saved as float WAV, preserving peaks for measurement. Playback is
  not loudness-normalized. Near/full-scale sample percentage is a diagnostic,
  not proof of audible clipping.
- Flash uses its fixed four-step/CFG-off recipe. Base uses 32 steps/CFG 2. The app
  does not send unsupported per-request sampling overrides. Launcher batch limits
  are conditioning=2, DiT=2, decode=1; the evaluation worker runs one request at a
  time so A/B timings do not include competition from other lab requests.
- Cancellation skips remaining takes. A running backend call is allowed to finish
  and its result is discarded, because closing HTTP does not reliably stop GPU
  inference. A restarted app marks unfinished jobs interrupted and does not replay
  them. An upstream request may outlive an app shutdown or network timeout.
- Prompt Enhancer and automatic ASR are **not** included. This lab exposes the
  model's direct instruction behavior; no cloud LLM credentials are needed.

## Measurements and reproducibility

Each result records its instruction/payload, source ID, seed, model variant,
requested/actual duration, latency, RTF, duration error, sample rate, peak/RMS,
near/full-scale samples, and review. JSON exports include source hashes and
provenance. Audio files remain in `data/`; export JSON is metadata, not an audio
archive. Back up the whole `data/` directory to retain recordings.

RTF = HTTP request elapsed time / generated seconds. This includes backend work
and network overhead, but excludes app queue time and upload preparation. Cold
inference and warm inference differ: run an initial warmup separately, then
compare identical clips, duration, and seeds. The lab does not calculate WER,
speaker embeddings, or GPU-memory peaks. Use `nvidia-smi`/host monitoring alongside
the listening scores; do not interpret manual scores as automatic quality metrics.

The Docker runtime image is pinned by digest, and SGLang-Omni plus all three HF
repositories are pinned in [lab/runtime.json](lab/runtime.json). The launcher
downloads those exact snapshots and explicitly sets the served model names.
SGLang source is installed without replacing the image's prebuilt CUDA stack.
The app's recorded runtime describes this supplied deployment; externally managed
endpoints must use matching versions if you want the same provenance.

## Existing SGLang-Omni endpoints / local development

The UI/API requires no Torch or CUDA. With Python 3.12 and FFmpeg:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AUK_FLASH_URL=http://127.0.0.1:8101
export AUK_BASE_URL=http://127.0.0.1:8102
export AUK_OFFICIAL_FLASH_URL=http://127.0.0.1:8201
export AUK_OFFICIAL_BASE_URL=http://127.0.0.1:8202
uvicorn lab.app:app --host 127.0.0.1 --port 7865 --workers 1
```

Use exactly one app worker: its serial job queue is process-local. External
services should advertise `tencent/AuK-Flash` / `tencent/AuK` through `/v1/models`
(CLI `--model-name`) and support the pinned AuK protocol. The app verifies identity
before sending a job, avoiding comparisons against accidentally swapped endpoints.

## Validation

```bash
pip install -r requirements-dev.txt
python -m pytest -q
node --check static/app.js
bash -n auk-lab.sh
docker compose --profile both config --quiet
```

After the GPU services report ready, run a real inference smoke test:

```bash
# Run from a Python environment with httpx installed:
python tools/smoke.py --model flash
python tools/smoke.py --model both
python tools/smoke.py --backend official --model flash
# Or use the running UI container, which already has httpx:
docker compose exec -T lab python - < tools/smoke.py
```

Current validation: CPU/API tests and JavaScript/Bash syntax checks pass; the
browser UI was inspected locally. Tests inject a mock SGLang transport and
explicit synthetic tones only inside tests. **Docker image build, real GPU
inference, VRAM use, and audio quality still require validation on the Linux
RTX PRO 6000 host.**

The official adapter is tested with an injected fake model only. The reported
server TTS success covers the original SGLang deployment; editing quality and
the new official Docker backend still need the on-server comparison.

## Project layout

- `lab/`: FastAPI routes, task catalog, protocol adapter, serial runner, artifacts.
- `static/`: standalone HTML/CSS/JavaScript listening interface.
- `docker/`: pinned SGLang-Omni image and GPU/checkpoint startup.
- `quickstart.sh`: environment setup, container builds, startup, readiness wait.
- `compose.yaml`, `auk-lab.sh`: Linux service lifecycle.
- `tests/`: protocol, upload, comparison, cancellation, persistence and export tests.
- `data/`: saved source clips, output WAVs, job manifests, and listening reviews.

References: [AuK source](https://github.com/Tencent-Hunyuan/AuK),
[model card](https://huggingface.co/tencent/AuK),
[task cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md),
[SGLang-Omni AuK API](https://sgl-project.github.io/sglang-omni/cookbook/auk.html).
