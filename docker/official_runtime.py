"""Import the official torch-attention path without optional FlashAttention.

The shared CUDA image can expose a flash_attn namespace without the legacy
functions AuK imports. AukInfer explicitly selects attn_backend="torch", so
hide that unused package during import instead of changing model source or
uninstalling packages from the shared platform.
"""
import importlib
import sys


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
