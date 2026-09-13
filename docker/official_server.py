"""Small HTTP adapter around the unmodified official AukInfer.generate API.

No SGLang model code, prompt enhancer, or audio post-effects are used here.
Only the lab's nonstreaming, mono 24 kHz subset of the API is implemented.
"""
import base64
import binascii
from contextlib import asynccontextmanager
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response


def parse_request(body, speech=False):
    if body.get('stream'):
        raise ValueError('Streaming is not supported')
    params = body if speech else body.get('metadata', {}).get('tts_params', {})
    ref = params.get('ref_audio')
    duration = body.get('stage_params', {}).get('auk_engine', {}).get('gen_seconds')
    if duration is None or not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or not 0.02 <= duration <= 30:
        raise ValueError('Set stage_params.auk_engine.gen_seconds between 0.02 and 30')
    seed = params.get('seed', 1234)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer between 0 and 2**32-1')
    if speech:
        if body.get('response_format', 'wav') != 'wav':
            raise ValueError('Only WAV responses are supported')
        text = body.get('input', '').strip()
        if not text:
            raise ValueError('Text to speak is required')
        description = (body.get('instructions') or '').strip() or 'A clear, natural voice.'
        instruction = (f'Say the following with the same voice: "{text}"' if ref else
                       f'Generate speech based on the following description: "{description}". '
                       f'The content to speak is: "{text}".')
    else:
        instruction = body.get('prompt', '').strip()
    if not instruction or len(instruction) > 14000:
        raise ValueError('A nonempty instruction of at most 14000 characters is required')
    audio = None
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith('data:audio/wav;base64,'):
            raise ValueError('ref_audio must be an inline WAV data URI; file paths and URLs are unsupported')
        if len(ref) > 8_000_000:
            raise ValueError('Reference audio is too large')
        try:
            data = base64.b64decode(ref.split(',', 1)[1], validate=True)
            wave, rate = sf.read(io.BytesIO(data), always_2d=True, dtype='float32')
        except (ValueError, RuntimeError, binascii.Error) as exc:
            raise ValueError('Invalid reference WAV') from exc
        if rate != 24000 or wave.shape[1] != 1 or not 480 <= len(wave) <= 720000 or not np.isfinite(wave).all():
            raise ValueError('Reference must be finite mono 24 kHz audio, 0.02–30 seconds')
        audio = wave[:, 0]
    return instruction, audio, float(duration), seed


def create_app(engine_factory=None, seed_fn=None, tensor_fn=None):
    variant = os.environ.get('AUK_VARIANT', 'flash')
    if variant not in {'flash', 'base'}:
        raise ValueError('AUK_VARIANT must be flash or base')
    model_id = 'tencent/AuK-Flash' if variant == 'flash' else 'tencent/AuK'
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = (engine_factory or load_engine)(variant)
        yield

    app = FastAPI(title='Official AuK Python adapter', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'backend': 'official', 'model': model_id}

    @app.get('/v1/models')
    def models():
        return {'data': [{'id': model_id, 'owned_by': 'tencent', 'object': 'model'}]}

    def generate(body, speech):
        try:
            if body.get('model', model_id) != model_id:
                raise ValueError('Model does not match this service')
            instruction, audio, duration, seed = parse_request(body, speech)
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        # Official code uses global RNG and a mutable transformer cache; serialize calls.
        with lock, tempfile.TemporaryDirectory(prefix='auk-official-') as directory:
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': instruction}]}]
            audio_arg = None
            if audio is not None:
                path = str(Path(directory) / 'reference.wav')
                sf.write(path, audio, 24000, subtype='FLOAT')
                messages[0]['content'].append({'type': 'audio', 'audio': path})
                if tensor_fn is None:
                    import torch
                    audio_arg = (torch.from_numpy(audio.copy()).unsqueeze(0), 24000)
                else:
                    audio_arg = (tensor_fn(audio), 24000)
            if seed_fn is None:
                import torch
                torch.manual_seed(seed)
            else:
                seed_fn(seed)
            engine = app.state.engine
            try:
                output, rate = engine.generate(messages, audio=audio_arg, gen_seconds=duration, seed=seed,
                                              nfe=int(os.environ.get('AUK_BASE_NFE', '32')),
                                              cfg_strength=float(os.environ.get('AUK_BASE_CFG', '2.0')))
                wave = output.detach().float().cpu().numpy().reshape(-1)
                if rate != 24000 or not len(wave) or len(wave) > 722400 or not np.isfinite(wave).all():
                    raise RuntimeError('Official model returned invalid audio')
                stream = io.BytesIO()
                sf.write(stream, wave, rate, format='WAV', subtype='FLOAT')
                data = stream.getvalue()
            finally:
                engine.model.transformer.clear_cache()
        if speech:
            return Response(data, media_type='audio/wav')
        return {'audio': {'data': base64.b64encode(data).decode(), 'format': 'wav'},
                'meta_info': {'finish_reason': {'type': 'stop'}, 'completion_tokens': len(wave)//480,
                              'weight_version': 'official-python'}}

    @app.post('/generate')
    def edit(body: dict):
        return generate(body, False)

    @app.post('/v1/audio/speech')
    def speech(body: dict):
        return generate(body, True)

    return app


def load_engine(variant):
    import torch
    from huggingface_hub import snapshot_download
    from auk.infer.infer_auk import AukInfer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; check NVIDIA Container Toolkit')
    runtime = json.loads(Path('/opt/auk-lab-runtime.json').read_text())
    def snapshot(spec):
        return Path(snapshot_download(repo_id=spec['id'], revision=spec['revision']))
    model = snapshot(runtime['models'][variant])
    encoder = snapshot(runtime['encoder'])
    weights = [p for p in model.glob('*.safetensors') if p.name != 'vae.safetensors']
    if len(weights) != 1:
        raise RuntimeError(f'Expected one AuK weight file, found {len(weights)}')
    print(json.dumps(dict(backend='official', runtime=runtime, gpu=torch.cuda.get_device_name(0),
                          torch=torch.__version__, cuda=torch.version.cuda, weight_dtype='float32',
                          autocast='bfloat16', prompt_enhancer=False)), flush=True)
    return AukInfer(str(model / 'config.yaml'), str(weights[0]), device='cuda:0', dtype='bf16', qwen_path=str(encoder))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=8000, workers=1)
