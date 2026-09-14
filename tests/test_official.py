"""Adapter/dispatch tests with explicit fake model output, never quality tests."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from docker.official_server import create_app, parse_request
from docker.serve import command
from lab.app import create_app as create_lab
from lab.client import BackendClient, build_request
from lab.schema import RunRequest
from tools.diagnose_edits import compare_audio
from test_lab import wav, wait_job


def test_reference_comparison_does_not_confuse_gain_with_identity():
    source = wav()
    assert compare_audio(source, source)['exact_samples_equal'] is True
    samples, rate = sf.read(io.BytesIO(source))
    stream = io.BytesIO()
    sf.write(stream, samples * 0.1, rate, format='WAV', subtype='FLOAT')
    result = compare_audio(source, stream.getvalue())
    assert result['rms_delta_db'] == pytest.approx(-20, abs=0.01)
    assert result['unaligned_cosine'] == pytest.approx(1)
    assert result['exact_samples_equal'] is False


def test_sglang_parity_storage_default(monkeypatch):
    monkeypatch.delenv('AUK_WEIGHT_DTYPE', raising=False)
    args = command()
    assert args[args.index('--auk_engine.factory.weight_dtype') + 1] == 'float32'
    monkeypatch.setenv('AUK_WEIGHT_DTYPE', 'bfloat16')
    args = command()
    assert args[args.index('--auk_engine.factory.weight_dtype') + 1] == 'bfloat16'


def test_official_adapter_calls_original_api_with_reference_and_clears_cache(monkeypatch):
    monkeypatch.setenv('AUK_VARIANT', 'flash')
    calls, seeds, cleared = [], [], []
    class Tensor:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.ones((1, 4800), dtype=np.float32) * .05
    class FakeEngine:
        model = SimpleNamespace(transformer=SimpleNamespace(clear_cache=lambda: cleared.append(True)))
        def generate(self, messages, **kwargs):
            assert Path(messages[0]['content'][1]['audio']).is_file()
            calls.append((messages, kwargs))
            return Tensor(), 24000
    app = create_app(lambda _: FakeEngine(), seed_fn=seeds.append, tensor_fn=lambda a: a)
    request = RunRequest(task='custom', source_id='a'*32, instruction="把'跑车'替换为'嘉年华'")
    route, payload, _ = build_request(request, 'flash', wav(), .2, 1234)
    with TestClient(app) as client:
        response = client.post(route, json=payload)
        assert response.status_code == 200, response.text
        assert base64.b64decode(response.json()['audio']['data'])[:4] == b'RIFF'
        assert client.get('/health').json()['backend'] == 'official'
    messages, kwargs = calls[0]
    assert messages[0]['content'][0]['text'] == request.instruction
    assert not Path(messages[0]['content'][1]['audio']).exists()
    assert kwargs['gen_seconds'] == .2 and kwargs['seed'] == 1234
    assert kwargs['audio'][1] == 24000 and len(kwargs['audio'][0]) == 4800
    assert kwargs['nfe'] == 32 and kwargs['cfg_strength'] == 2
    assert seeds == [1234] and cleared == [True]


def test_official_speech_templates_and_invalid_reference():
    speech = dict(input='Hello', instructions='Warm', stage_params={'auk_engine': {'gen_seconds': 1}})
    instruction, source, _, _ = parse_request(speech, True)
    assert instruction == 'Generate speech based on the following description: "Warm". The content to speak is: "Hello".'
    speech['ref_audio'] = 'data:audio/wav;base64,' + base64.b64encode(wav()).decode()
    assert parse_request(speech, True)[0] == 'Say the following with the same voice: "Hello"'
    speech['ref_audio'] = '/etc/passwd'
    with pytest.raises(ValueError, match='inline WAV'):
        parse_request(speech, True)


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_inference_error_reaches_client_without_cleanup_masking_it(cleanup_fails, caplog):
    def clear_cache():
        if cleanup_fails:
            raise RuntimeError('secondary cleanup failure')
    class BrokenEngine:
        model = SimpleNamespace(transformer=SimpleNamespace(clear_cache=clear_cache))
        def generate(self, *args, **kwargs):
            raise RuntimeError('test reference encoder failure')
    app = create_app(lambda _: BrokenEngine(), seed_fn=lambda _: None, tensor_fn=lambda a: a)
    request = RunRequest(task='custom', source_id='a'*32, instruction='Change the words')
    route, payload, _ = build_request(request, 'flash', wav(), .2, 1234)
    with TestClient(app) as client:
        response = client.post(route, json=payload)
        assert response.status_code == 500
        detail = response.json()['detail']
        assert 'RuntimeError: test reference encoder failure' in detail
        assert 'secondary cleanup failure' not in detail
        assert '[error ' in detail
        assert 'test reference encoder failure' in caplog.text
        assert 'data:audio/wav' not in caplog.text


@pytest.mark.parametrize('task', ['voice_design', 'custom'])
def test_backend_comparison_dispatch_and_provenance(tmp_path, task):
    calls = []
    def handle(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'backend': 'official' if request.url.host == 'official' else 'sglang'})
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'tencent/AuK-Flash'}]})
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        if request.url.path == '/generate':
            return httpx.Response(200, json={'audio': {'data': base64.b64encode(wav()).decode(), 'format': 'wav'}})
        return httpx.Response(200, content=wav())
    backend = BackendClient({'flash': 'http://sglang', 'official_flash': 'http://official'}, transport=httpx.MockTransport(handle))
    with TestClient(create_lab(tmp_path, backend)) as client:
        request = dict(task=task, backend='both', models=['flash'], text='Hello', instruction='Warm', duration=.2)
        if task == 'custom':
            source = client.post('/api/sources', files={'file': ('reference.wav', wav(), 'audio/wav')})
            assert source.status_code == 200, source.text
            request.update(source_id=source.json()['id'], instruction="把'跑车'替换为'嘉年华'")
        response = client.post('/api/jobs', json=request)
        assert response.status_code == 202, response.text
        job = wait_job(client, response.json()['id'])
        assert job['status'] == 'completed', job
        assert [r['backend'] for r in job['results']] == ['sglang', 'official']
        assert calls[0][0] == 'sglang' and calls[1][0] == 'official'
        assert calls[0][1] == calls[1][1]
        if task == 'custom':
            assert calls[0][1]['prompt'] == request['instruction']
            assert calls[0][1]['metadata']['tts_params']['ref_audio'].startswith('data:audio/wav;base64,')
        assert job['results'][1]['sglang_revision'] is None
        assert job['results'][1]['runtime']['official_revision']
        assert 'backend' in client.get('/api/export?format=csv').text.splitlines()[0]


def test_official_endpoint_rejects_mislabeled_sglang():
    import asyncio
    async def check():
        backend = BackendClient({'official_flash': 'http://wrong'}, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={'status': 'healthy'})))
        try:
            assert (await backend.status('official_flash'))['ready'] is False
        finally:
            await backend.close()
    asyncio.run(check())
