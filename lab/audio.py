import asyncio
import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf

MAX_SECONDS = 30


def describe_audio(data: bytes) -> dict:
    try:
        wave, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("The audio response could not be decoded") from exc
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    peak = float(np.abs(wave).max())
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return dict(duration=round(len(wave) / rate, 6), sample_rate=rate, channels=wave.shape[1],
                peak_dbfs=round(20 * math.log10(max(peak, 1e-10)), 2),
                rms_dbfs=round(20 * math.log10(max(rms, 1e-10)), 2),
                clipped_percent=round(float(np.mean(np.abs(wave) >= 0.999)) * 100, 4))


def normalize_result(data: bytes) -> bytes:
    describe_audio(data)
    wave, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate != 24000 or wave.shape[1] != 1:
        raise ValueError(f"Expected AuK 24 kHz mono audio, received {rate} Hz / {wave.shape[1]} channels")
    if len(wave) / rate > MAX_SECONDS + 0.1:
        raise ValueError("Backend returned audio longer than the lab's 30-second limit")
    output = io.BytesIO()
    # Float WAV preserves peaks for evaluation instead of clipping them to PCM16.
    sf.write(output, wave, rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


async def prepare_upload(source: Path, target: Path, start: float, end: float | None) -> dict:
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError("Crop end must be greater than start; times must be finite and nonnegative")
    if end is not None and end - start > MAX_SECONDS:
        raise ValueError("Select at most 30 seconds of source audio")
    # Decode only 31 seconds; a longer uncropped source is rejected, never silently truncated.
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
            "-ss", str(start), "-i", str(source), "-t", str(end - start if end is not None else 31),
            "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_f32le", str(target)]
    try:
        process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.DEVNULL,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ValueError("FFmpeg is required to decode uploads") from exc
    try:
        _, error = await asyncio.wait_for(process.communicate(), timeout=60)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise ValueError("FFmpeg could not decode this clip: " + error.decode(errors="replace")[-400:])
    info = describe_audio(target.read_bytes())
    if info["duration"] > MAX_SECONDS + 0.001:
        raise ValueError("Source is longer than 30 seconds. Set crop start and end before uploading")
    if info["duration"] < 0.02:
        raise ValueError("Source must contain at least 20 ms of audio")
    return info
