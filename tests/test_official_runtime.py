"""Reproduce an incomplete flash_attn namespace without installing Torch/CUDA."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("preloaded", [False, True])
def test_import_skips_incomplete_flash_attention_and_preserves_modules(tmp_path, preloaded):
    # Match the user's image: discoverable namespace, absent flash_attn_func.
    (tmp_path / "flash_attn").mkdir()
    package = tmp_path / "auk" / "infer"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "infer_auk.py").write_text(
        "import importlib.util\n"
        "if importlib.util.find_spec('flash_attn') is not None:\n"
        "    from flash_attn import flash_attn_func, flash_attn_varlen_func\n"
        "class AukInfer: pass\n", encoding="utf-8")
    script = f"""
import importlib, importlib.util, sys
from docker.official_runtime import load_auk_infer
assert importlib.util.find_spec('flash_attn') is not None
try:
    importlib.import_module('auk.infer.infer_auk')
except ImportError as error:
    assert 'flash_attn_func' in str(error), error
else:
    raise AssertionError('Expected original import failure')
previous = importlib.import_module('flash_attn') if {preloaded!r} else sys.modules.pop('flash_attn', None)
klass = load_auk_infer()
assert klass.__name__ == 'AukInfer'
assert sys.modules['auk.infer.infer_auk'].AukInfer is klass
assert load_auk_infer() is klass
if {preloaded!r}:
    assert sys.modules['flash_attn'] is previous
else:
    assert 'flash_attn' not in sys.modules
"""
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(root)])}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_errors_are_not_swallowed_and_previous_module_is_restored(monkeypatch):
    from docker import official_runtime
    previous = object()
    monkeypatch.setitem(sys.modules, "flash_attn", previous)
    def fail(name):
        assert sys.modules['flash_attn'] is None
        raise ImportError('unrelated dependency failed')
    monkeypatch.setattr(official_runtime.importlib, "import_module", fail)
    with pytest.raises(ImportError, match='unrelated dependency failed'):
        official_runtime.load_auk_infer()
    assert sys.modules['flash_attn'] is previous
