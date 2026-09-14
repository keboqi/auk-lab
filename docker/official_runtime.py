"""Import the official torch-attention path without optional FlashAttention.

The shared CUDA image can expose a flash_attn namespace without the legacy
functions AuK imports. AukInfer explicitly selects attn_backend="torch", so
hide that unused package during import instead of changing model source or
uninstalling packages from the shared platform.
"""
import importlib
import sys


def check_reference_audio():
    """Exercise Qwen's real CPU reference loader before loading GPU weights.

    The temporary tone is a dependency probe only, never a generated result.
    qwen-omni-utils 0.0.9 imports audioread but omits it from its requirements.
    """
    from pathlib import Path
    import tempfile
    import numpy as np
    import soundfile as sf
    from qwen_omni_utils import process_mm_info

    with tempfile.TemporaryDirectory(prefix="auk-dependency-check-") as directory:
        path = Path(directory) / "reference.wav"
        wave = (0.1 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24000)).astype(np.float32)
        sf.write(path, wave, 24000, subtype="FLOAT")
        messages = [[{"role": "user", "content": [
            {"type": "text", "text": "Keep the speech unchanged."},
            {"type": "audio", "audio": str(path)},
        ]}]]
        audios, _, _ = process_mm_info(messages, use_audio_in_video=True)
        if audios is None or len(audios) != 1:
            raise RuntimeError("Qwen reference-audio dependency check returned no audio")
        decoded = np.asarray(audios[0])
        if decoded.shape != (1600,) or not np.isfinite(decoded).all() or not np.any(decoded):
            raise RuntimeError("Qwen reference-audio dependency check failed 24 kHz to 16 kHz decoding")
    print("Qwen reference-audio dependency check passed (24 kHz WAV -> 16 kHz mono)", flush=True)


def load_auk_infer():
    missing = object()
    previous = sys.modules.get("flash_attn", missing)
    sys.modules["flash_attn"] = None
    try:
        return importlib.import_module("auk.infer.infer_auk").AukInfer
    finally:
        # Restore only this entry; retain all modules loaded by AuK's import.
        if previous is missing:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = previous
