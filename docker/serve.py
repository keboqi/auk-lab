"""Start a pinned AuK pipeline with conservative, explicit Blackwell settings."""
import json
import os
import subprocess
import sys
from pathlib import Path


def command(model_path=None, encoder_path=None):
    variant = os.environ.get("AUK_VARIANT", "flash")
    if variant not in {"base", "flash"}:
        raise ValueError("AUK_VARIANT must be base or flash")
    model_id = "tencent/AuK-Flash" if variant == "flash" else "tencent/AuK"
    args = [sys.executable, "-m", "sglang_omni.cli", "serve",
            "--model-path", model_path or model_id, "--model-name", model_id,
            "--host", "0.0.0.0", "--port", "8000",
            "--preprocessing.factory.max_seconds", "30",
            "--conditioning.factory.max_batch_size", os.environ.get("AUK_CONDITIONING_BATCH", "2"),
            "--auk_engine.factory.max_batch_size", os.environ.get("AUK_DIT_BATCH", "2"),
            "--decode.factory.max_batch_size", os.environ.get("AUK_DECODE_BATCH", "1"),
            "--auk_engine.factory.max_seconds", "30",
            "--auk_engine.factory.weight_dtype", os.environ.get("AUK_WEIGHT_DTYPE", "bfloat16"),
            "--auk_engine.factory.nfe", os.environ.get("AUK_BASE_NFE", "32"),
            "--auk_engine.factory.cfg_strength", os.environ.get("AUK_BASE_CFG", "2.0")]
    if encoder_path:
        args.extend(["--conditioning.factory.text_encoder_path", encoder_path])
    return args


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check NVIDIA Container Toolkit and Docker GPU access.")
    props = torch.cuda.get_device_properties(0)
    info = dict(gpu=props.name, total_gib=round(props.total_memory / 1024**3, 2),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                torch=torch.__version__, cuda=torch.version.cuda, arch_list=torch.cuda.get_arch_list())
    print("AuK Lab device: " + json.dumps(info), flush=True)
    if props.total_memory < 80 * 1024**3:
        print("This launcher targets the 96 GB RTX PRO 6000. Use one model at a time on smaller GPUs.", flush=True)
    # Exercise a CUDA kernel before starting several expensive model loads.
    print("CUDA kernel check:", (torch.ones(16, device="cuda") + 1).sum().item(), flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    from huggingface_hub import snapshot_download
    runtime = json.loads(Path("/opt/auk-lab-runtime.json").read_text())
    model = runtime["models"][os.environ.get("AUK_VARIANT", "flash")]
    encoder = runtime["encoder"]
    print("Downloading/reusing pinned checkpoints:", json.dumps(runtime), flush=True)
    model_path = snapshot_download(repo_id=model["id"], revision=model["revision"])
    encoder_path = snapshot_download(repo_id=encoder["id"], revision=encoder["revision"])
    args = command(model_path, encoder_path)
    print("Starting:", " ".join(args), flush=True)
    os.execv(sys.executable, args)
