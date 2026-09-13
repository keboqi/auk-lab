"""Exercise bootstrap behavior with fake host tools; never start Docker here."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bash_path():
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    path = str(candidate) if candidate.is_file() else shutil.which("bash")
    if not path:
        pytest.skip("Bash not available")
    return path


def test_bootstrap_defaults_overrides_and_preserves_environment(tmp_path):
    for name in ("quickstart.sh", "auk-lab.sh", ".env.example"):
        shutil.copy(ROOT / name, tmp_path / name)
    fake = tmp_path / "fake"
    fake.mkdir()
    scripts = {
        "uname": "#!/bin/bash\necho Linux\n",
        "nvidia-smi": "#!/bin/bash\necho '0, RTX PRO 6000 Blackwell, 96 GB'\n",
        "docker": "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$AUK_TEST_LOG\"\nif [[ \"$*\" == 'compose port lab 7865' ]]; then echo 127.0.0.1:7865; fi\nif [[ \"$*\" == 'compose exec -T lab python - '* ]]; then cat >/dev/null; fi\nexit 0\n",
    }
    for name, body in scripts.items():
        path = fake / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    env = {**os.environ, "AUK_TEST_LOG": (tmp_path / "calls.txt").as_posix()}
    # Let Bash translate the absolute Windows argument before constructing PATH.
    invoke = [bash_path(), "-c", 'export PATH="$(cd "$1" && pwd):$PATH"; shift; bash quickstart.sh "$@"', "test", str(fake)]
    result = subprocess.run(invoke, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AuK Lab is ready" in result.stdout
    assert (tmp_path / ".env").read_text() == (tmp_path / ".env.example").read_text()
    calls = (tmp_path / "calls.txt").read_text()
    assert "compose --profile flash up -d --build" in calls
    assert "compose --profile both stop auk-base" in calls
    (tmp_path / ".env").write_text("LAB_PORT=9999\n", encoding="utf-8")
    result = subprocess.run(invoke + ["--model", "both", "--gpu", "0", "--port", "7870", "--no-wait"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / ".env").read_text() == "LAB_PORT=9999\n"
    assert "compose --profile both up -d --build" in (tmp_path / "calls.txt").read_text()


@pytest.mark.parametrize("args", [["--model", "unknown"], ["--port", "99999"], ["--gpu", "x"], ["--model"]])
def test_bad_bootstrap_arguments_fail_before_host_changes(args):
    result = subprocess.run([bash_path(), str(ROOT / "quickstart.sh"), *args], capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 2
