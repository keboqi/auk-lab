"""CPU-only inspection of the installed backend's actual request/processor path.

Read one unredacted /generate request from stdin. Never print reference bytes.
This checks preprocessing, not live worker hidden states or model quality.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import Qwen2_5OmniProcessor
    from sglang_omni.client.client import Client
    from sglang_omni.models.auk.hf_config import make_runtime_config
    from sglang_omni.models.auk.reference_encode import build_messages
    from sglang_omni.models.auk.request_builders import build_auk_state
    from sglang_omni.proto import StagePayload
    from sglang_omni.serve.openai_api import _build_rollout_generate_request
    from sglang_omni.serve.protocol import RolloutGenerateRequest

    raw = json.load(sys.stdin)
    runtime = json.loads(Path('/opt/auk-lab-runtime.json').read_text())
    variant = os.environ.get('AUK_VARIANT', 'flash')
    def snapshot(spec):
        return snapshot_download(repo_id=spec['id'], revision=spec['revision'], local_files_only=True)
    request = _build_rollout_generate_request(RolloutGenerateRequest(**raw))
    payload = StagePayload(request_id='edit-diagnostic', request=Client._build_omni_request(request), data={})
    state = build_auk_state(payload, make_runtime_config(snapshot(runtime['models'][variant])))
    assert state.instruction == raw['prompt'], 'Instruction changed during preprocessing'
    assert state.ref_audio is not None and state.qwen_audio is not None, 'Missing reference audio'
    processor = Qwen2_5OmniProcessor.from_pretrained(snapshot(runtime['encoder']))
    messages = build_messages(state.instruction, True)
    formatted = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=formatted, audio=[state.qwen_audio], padding=True, return_tensors='pt')
    assert 'input_features' in inputs, 'Qwen did not produce audio features'
    report = dict(check='installed_backend_preprocessing', model=variant, runtime=runtime,
                  weight_dtype=os.environ.get('AUK_WEIGHT_DTYPE', 'float32'),
                  instruction=state.instruction, chat_template=formatted,
                  seed=state.seed, gen_frames=state.gen_frames,
                  reference_samples=len(state.ref_audio), qwen_samples=len(state.qwen_audio),
                  reference_sha256=hashlib.sha256(np.asarray(state.ref_audio).tobytes()).hexdigest(),
                  processor_shapes={k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')},
                  prompt_tokens=int(inputs['attention_mask'].sum()),
                  limitation='Checks the installed preprocessing code; does not inspect live worker tensors or prove editing quality.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
