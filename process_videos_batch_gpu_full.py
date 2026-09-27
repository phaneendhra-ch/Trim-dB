"""
GPU-accelerated batch video processor (enhanced).

- Uses hardware-accelerated encoders when available (NVIDIA/AMD/Intel)
- Scans audio in a single pass (ffmpeg WAV + numpy)
- Keeps the same dB / subclip logic as the old MoviePy script
- Cuts with input seeks + small concat graphs (never one giant select expression)
- Encodes each batch with NVENC, then concatenates
"""

import argparse
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# GPU codecs in priority order (NVIDIA typically fastest for encoding)
GPU_CODECS = [
    ("h264_nvenc", "NVIDIA GPU"),
    ("h264_amf", "AMD GPU"),
    ("h264_qsv", "Intel QuickSync"),
]

INTERVAL = 0.5
THRESHOLD_DB_MAX = -46.0
# Sample rate for dB analysis only (not used in the output video)
DB_ANALYSIS_SR = 16000
# How many keep-ranges to feed as separate -ss/-i inputs per ffmpeg process.
# A 985-term select() graph OOMs; this stays small and uses GPU encode.
INPUT_BATCH_SIZE = 8


def log(msg: str = "") -> None:
    print(msg, flush=True)


def detect_gpu_encoder() -> Tuple[Optional[str], str]:
    """
    Detect available hardware H.264 encoder via ffmpeg.
    Returns (codec_name, description) or (None, "CPU fallback") if none found.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        encoders_output = result.stdout + result.stderr

        for codec, desc in GPU_CODECS:
            for line in encoders_output.splitlines():
                if line.strip().startswith("V") and codec in line:
                    return codec, desc

        return None, "CPU (no GPU encoder found)"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None, "CPU (ffmpeg check failed)"


def encoder_works(video_codec: str) -> bool:
    """Sanity-check that ffmpeg can actually encode a few frames with this codec."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x144:d=0.2:r=30",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo:d=0.2",
        "-c:v",
        video_codec,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        "0.2",
        "-f",
        "null",
        "-",
    ]
    if video_codec == "h264_nvenc":
        insert_at = cmd.index("-c:v") + 2
        cmd[insert_at:insert_at] = ["-preset", "p4", "-gpu", "0"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def encoder_args(video_codec: str, use_gpu: bool) -> List[str]:
    args = [
        "-c:v",
        video_codec,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-b:a",
        "160k",
    ]
    if not use_gpu:
        args.extend(["-preset", "veryfast"])
        return args
    if video_codec == "h264_nvenc":
        # Encode on NVENC (Video Encode engine), not CUDA 3D / system RAM.
        args.extend(
            [
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-b:v",
                "0",
                "-profile:v",
                "high",
                "-g",
                "60",
                "-bf",
                "2",
                "-gpu",
                "0",
            ]
        )
    elif video_codec == "h264_qsv":
        args.extend(["-preset", "veryfast"])
    elif video_codec == "h264_amf":
        args.extend(["-quality", "speed"])
    return args


def decode_hwaccel_args(video_codec: str, use_gpu: bool) -> List[str]:
    """Optional GPU decode. Filters still run on CPU; encoder stays on GPU."""
    if not use_gpu:
        return []
    if video_codec == "h264_nvenc":
        return ["-hwaccel", "cuda"]
    if video_codec == "h264_qsv":
        return ["-hwaccel", "qsv"]
    if video_codec == "h264_amf":
        return ["-hwaccel", "d3d11va"]
    return []


def ffprobe_duration(path: str) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def ffprobe_fps_fraction(path: str) -> str:
    """
    Constant frame rate for the output (avoids frozen video after concat/select).
    Prefer r_frame_rate, then avg_frame_rate.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        for raw in lines:
            if "/" in raw:
                num_s, den_s = raw.split("/", 1)
                num, den = int(num_s), int(den_s)
                if den != 0 and num > 0:
                    return f"{num}/{den}"
            else:
                val = float(raw)
                if val > 0:
                    return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "30/1"


def _parse_progress_seconds(line: str) -> Optional[float]:
    line = line.strip()
    if "=" not in line:
        return None
    key, _, raw = line.partition("=")
    raw = raw.strip()
    if raw in ("", "N/A"):
        return None
    try:
        if key == "out_time_ms":
            return int(raw) / 1_000_000.0
        if key == "out_time_us":
            return int(raw) / 1_000_000.0
        if key == "out_time":
            # HH:MM:SS.microseconds
            parts = raw.split(":")
            if len(parts) != 3:
                return None
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None
    return None


def _ffmpeg_error_text(stderr_path: Path) -> str:
    try:
        err = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    keys = ("error", "failed", "invalid", "cannot", "not found", "unrecognized")
    interesting = [
        ln for ln in err.splitlines() if any(k in ln.lower() for k in keys)
    ]
    if interesting:
        return "\n".join(interesting[-12:]).strip()
    return err[-1200:].strip()


def run_ffmpeg_with_progress(
    cmd: List[str],
    total_seconds: Optional[float],
    label: str,
) -> None:
    """
    Run ffmpeg with -progress on stdout. stderr is written to a temp file so the
    process cannot deadlock on a full stderr pipe.
    """
    stderr_path = Path(tempfile.gettempdir()) / f"ffmpeg_{int(time.time() * 1000)}.log"
    last_report_pct = -1.0
    last_heartbeat = time.perf_counter()
    t0 = time.perf_counter()
    stats = {"frame": "?", "speed": "?", "out_s": 0.0}
    try:
        with open(stderr_path, "w", encoding="utf-8", errors="replace") as errf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=errf,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if line.startswith("frame="):
                    stats["frame"] = line.split("=", 1)[1]
                elif line.startswith("speed="):
                    stats["speed"] = line.split("=", 1)[1]
                seconds = _parse_progress_seconds(line)
                now = time.perf_counter()
                if seconds is not None:
                    stats["out_s"] = seconds
                pct = None
                if total_seconds and total_seconds > 0 and seconds is not None:
                    pct = min(100.0, 100.0 * seconds / total_seconds)
                    if pct - last_report_pct >= 1.0 or pct >= 100.0:
                        log(
                            f"  {label}: {pct:5.1f}%  "
                            f"{seconds:.1f}s / {total_seconds:.1f}s  "
                            f"frame={stats['frame']} speed={stats['speed']}  "
                            f"elapsed {now - t0:.0f}s"
                        )
                        last_report_pct = pct
                        last_heartbeat = now
                        continue
                if now - last_heartbeat >= 10:
                    extra = f"{pct:5.1f}%" if pct is not None else "working"
                    log(
                        f"  {label}: {extra}  "
                        f"frame={stats['frame']} speed={stats['speed']}  "
                        f"elapsed {now - t0:.0f}s (still running)"
                    )
                    last_heartbeat = now
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {rc}): {_ffmpeg_error_text(stderr_path)}"
            )
    finally:
        try:
            stderr_path.unlink(missing_ok=True)
        except OSError:
            pass


def extract_audio_wav(input_file: str, wav_path: str, source_duration: Optional[float]) -> None:
    """Extract 16 kHz mono PCM for fast dB analysis."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        input_file,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(DB_ANALYSIS_SR),
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]
    run_ffmpeg_with_progress(cmd, source_duration, "audio extract")


def compute_db_keep_sections(
    wav_path: str,
    interval: float = INTERVAL,
    threshold_db: float = THRESHOLD_DB_MAX,
) -> List[Tuple[float, float]]:
    """
    Single-pass RMS dB over `interval` windows. Adjacent keep windows are merged.
    Same keep rule as the old MoviePy script (keep if dB > threshold).
    """
    with wave.open(wav_path, "rb") as wav:
        nchannels, _sampwidth, framerate, nframes = wav.getparams()[:4]
        raw = wav.readframes(nframes)

    if not raw:
        return []

    samples = np.frombuffer(raw, dtype=np.int16)
    if nchannels > 1:
        samples = samples.reshape(-1, nchannels).mean(axis=1)
    samples = samples.astype(np.float32) / 32768.0
    n_samples = len(samples)
    if n_samples == 0 or framerate <= 0:
        return []

    duration = n_samples / float(framerate)
    samples_per_win = max(1, int(round(framerate * interval)))
    n_windows = int(np.ceil(n_samples / samples_per_win))

    keep_blocks: List[Tuple[float, float]] = []
    report_every = max(1, n_windows // 20)

    for i in range(n_windows):
        start_idx = i * samples_per_win
        end_idx = min(start_idx + samples_per_win, n_samples)
        if start_idx >= end_idx:
            continue
        chunk = samples[start_idx:end_idx]
        rms = float(np.sqrt(np.mean(chunk**2)))
        db = 20.0 * np.log10(rms) if rms > 0 else -np.inf
        start_time = start_idx / float(framerate)
        end_time = min(end_idx / float(framerate), duration)
        if db > threshold_db:
            keep_blocks.append((start_time, end_time))

        if i == 0 or (i + 1) % report_every == 0 or i + 1 == n_windows:
            log(
                f"  dB scan: window {i + 1}/{n_windows} "
                f"({start_time:.1f}s / {duration:.1f}s)"
            )

    if not keep_blocks:
        return []

    merged = [list(keep_blocks[0])]
    for start, end in keep_blocks[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s[0], s[1]) for s in merged]


def build_batch_concat_script(n: int, fps_frac: str) -> str:
    """Reset timestamps on each input clip, then concat — same idea as MoviePy."""
    lines: List[str] = []
    for i in range(n):
        lines.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={fps_frac},format=yuv420p[v{i}];"
        )
        lines.append(
            f"[{i}:a]asetpts=PTS-STARTPTS,"
            f"aresample=48000:async=1000:first_pts=0[a{i}];"
        )
    if n == 1:
        lines.append("[v0]setpts=PTS-STARTPTS[v];")
        lines.append("[a0]asetpts=PTS-STARTPTS[a]")
        return "\n".join(lines)
    joined = "".join(f"[v{i}][a{i}]" for i in range(n))
    lines.append(f"{joined}concat=n={n}:v=1:a=1[v][a]")
    return "\n".join(lines)


def _chunks(
    items: List[Tuple[float, float]], size: int
) -> List[List[Tuple[float, float]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _run_encode_cmd(cmd: List[str], total_seconds: float, label: str) -> None:
    run_ffmpeg_with_progress(cmd, total_seconds, label)


def write_concat_list(paths: List[Path], list_path: Path) -> None:
    lines = []
    for p in paths:
        posix = p.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{posix}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trimmed_video(
    input_file: str,
    output_file: str,
    keep_sections: List[Tuple[float, float]],
    keep_duration: float,
    video_codec: str,
    use_gpu: bool,
    work_dir: str,
) -> None:
    """
    Encode keep-ranges in small seek batches (GPU), then concat.

    Avoids ffmpeg select() graphs with hundreds of between() terms, which
    fail with 'Cannot allocate memory' on long recordings.
    """
    fps_frac = ffprobe_fps_fraction(input_file)
    log(f"  Output frame rate: {fps_frac}")
    batches = _chunks(keep_sections, INPUT_BATCH_SIZE)
    log(
        f"  Encoding {len(keep_sections)} keep range(s) in "
        f"{len(batches)} GPU batch(es) of up to {INPUT_BATCH_SIZE}"
    )

    work = Path(work_dir)
    batch_files: List[Path] = []
    hwaccel_ok = use_gpu

    for bi, batch in enumerate(batches, 1):
        batch_dur = sum(max(0.0, e - s) for s, e in batch)
        out_path = (
            Path(output_file)
            if len(batches) == 1
            else work / f"batch_{bi:04d}.mp4"
        )
        script_path = work / f"batch_{bi:04d}_filter.txt"
        script_path.write_text(
            build_batch_concat_script(len(batch), fps_frac), encoding="utf-8"
        )

        def make_cmd(
            with_hwaccel: bool,
            *,
            clips: List[Tuple[float, float]] = batch,
            filt: Path = script_path,
            dest: Path = out_path,
        ) -> List[str]:
            cmd: List[str] = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-nostats",
                "-progress",
                "pipe:1",
            ]
            for start, end in clips:
                dur = max(end - start, 0.04)
                if with_hwaccel:
                    cmd.extend(decode_hwaccel_args(video_codec, use_gpu))
                cmd.extend(
                    [
                        "-ss",
                        f"{start:.6f}",
                        "-t",
                        f"{dur:.6f}",
                        "-i",
                        input_file,
                    ]
                )
            cmd.extend(
                [
                    "-/filter_complex",
                    str(filt),
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    *encoder_args(video_codec, use_gpu),
                    "-fps_mode",
                    "cfr",
                    "-video_track_timescale",
                    "90000",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    str(dest),
                ]
            )
            return cmd

        label = f"encode batch {bi}/{len(batches)} ({video_codec})"
        encoded = False
        attempts: List[bool] = [hwaccel_ok] if hwaccel_ok else [False]
        if hwaccel_ok:
            attempts.append(False)
        last_error: Optional[Exception] = None
        for with_hw in attempts:
            try:
                _run_encode_cmd(make_cmd(with_hw), batch_dur, label)
                encoded = True
                if with_hw:
                    hwaccel_ok = True
                elif hwaccel_ok:
                    hwaccel_ok = False
                    log("  GPU decode failed on this batch; using CPU decode for later batches.")
                break
            except RuntimeError as exc:
                last_error = exc
                if with_hw:
                    log("  GPU decode failed, retrying this batch with CPU decode...")
                    continue
                log(f"  {label} failed: {exc}")
        if not encoded:
            assert last_error is not None
            raise last_error
        batch_files.append(out_path)

    if len(batch_files) == 1:
        return

    list_path = work / "concat.txt"
    write_concat_list(batch_files, list_path)
    log(f"  Concatenating {len(batch_files)} encoded batches...")
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-progress",
        "pipe:1",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_file,
    ]
    try:
        run_ffmpeg_with_progress(concat_cmd, keep_duration, "concat")
    except RuntimeError as exc:
        log(f"  Stream-copy concat failed ({exc}); re-encoding join with {video_codec}...")
        concat_re = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-progress",
            "pipe:1",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            *encoder_args(video_codec, use_gpu),
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            output_file,
        ]
        run_ffmpeg_with_progress(concat_re, keep_duration, f"concat re-encode ({video_codec})")


def process_video(
    input_file: str,
    output_file: str,
    video_codec: str = "libx264",
    use_gpu: bool = False,
) -> bool:
    """
    Process a single video file: trim silent/low-audio sections and save to output.
    """
    tmpdir = None
    t0 = time.perf_counter()
    try:
        source_duration = ffprobe_duration(input_file)
        if source_duration:
            log(f"  Source duration: {source_duration:.1f}s")

        log("  Extracting audio for dB scan...")
        tmpdir = tempfile.TemporaryDirectory()
        wav_path = str(Path(tmpdir.name) / "audio.wav")
        t_audio = time.perf_counter()
        extract_audio_wav(input_file, wav_path, source_duration)
        log(f"  Audio extracted in {time.perf_counter() - t_audio:.1f}s")

        log(f"  Scanning dB every {INTERVAL}s (keep above {THRESHOLD_DB_MAX} dB)...")
        t_scan = time.perf_counter()
        keep_sections = compute_db_keep_sections(wav_path)
        log(
            f"  dB scan done in {time.perf_counter() - t_scan:.1f}s — "
            f"{len(keep_sections)} keep segment(s) after merge"
        )

        if not keep_sections:
            log(f"  Warning: No audio above threshold in {input_file}, skipping.")
            return False

        keep_dur = sum(end - start for start, end in keep_sections)
        log(f"  Keep duration: {keep_dur:.1f}s")

        log(f"  Encoding {len(keep_sections)} segment(s) with {video_codec}...")
        t_enc = time.perf_counter()
        write_trimmed_video(
            input_file,
            output_file,
            keep_sections,
            keep_dur,
            video_codec,
            use_gpu,
            tmpdir.name,
        )
        log(f"  Encode done in {time.perf_counter() - t_enc:.1f}s")
        log(f"  File total: {time.perf_counter() - t0:.1f}s")
        return True
    except Exception as e:
        log(f"  Error processing {input_file}: {e}")
        return False
    finally:
        if tmpdir is not None:
            try:
                tmpdir.cleanup()
            except Exception:
                pass


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "GPU-accelerated batch video processor (enhanced): "
            "trim silent sections and save to New_Videos/ using GPU where ffmpeg allows."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing .mp4 files",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU encoding (libx264) instead of GPU",
    )
    args = parser.parse_args()

    log(f"Input: {args.input_dir}")
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        log(f"Error: Path is not a directory: {input_dir}")
        sys.exit(1)

    if args.cpu:
        video_codec = "libx264"
        use_gpu = False
        log("Using CPU encoding (libx264)\n")
    else:
        log("Checking ffmpeg GPU encoders...")
        gpu_codec, gpu_desc = detect_gpu_encoder()
        if gpu_codec:
            log(f"Found {gpu_codec} ({gpu_desc}), verifying a test encode...")
            if encoder_works(gpu_codec):
                video_codec = gpu_codec
                use_gpu = True
                log(f"Using GPU encoding: {gpu_codec} ({gpu_desc})")
                log(
                    "Note: Task Manager 'GPU 3D' may stay low; NVENC shows under "
                    "'Video Encode'. RAM should stay modest (no per-cut frame buffers).\n"
                )
            else:
                video_codec = "libx264"
                use_gpu = False
                log(
                    f"{gpu_codec} is listed but a test encode failed; "
                    "falling back to CPU (libx264)\n"
                )
        else:
            video_codec = "libx264"
            use_gpu = False
            log("Falling back to CPU encoding (libx264)\n")

    output_dir = input_dir / "New_Videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    mp4_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    )

    if not mp4_files:
        log(f"No .mp4 files found in {input_dir}")
        sys.exit(0)

    log(f"Found {len(mp4_files)} .mp4 file(s) to process\n")

    success_count = 0
    skipped_count = 0
    for i, input_path in enumerate(mp4_files, 1):
        output_path = output_dir / input_path.name
        if output_path.is_file() and output_path.stat().st_size > 0:
            skipped_count += 1
            log(
                f"[{i}/{len(mp4_files)}] Skipping (already in New_Videos): "
                f"{input_path.name}\n"
            )
            continue
        log(f"[{i}/{len(mp4_files)}] Processing: {input_path.name}")
        if process_video(
            str(input_path),
            str(output_path),
            video_codec=video_codec,
            use_gpu=use_gpu,
        ):
            success_count += 1
            log(f"  -> Saved: {output_path}\n")
        else:
            log("  -> Failed\n")

    log(
        f"Done. Processed {success_count}, skipped {skipped_count}, "
        f"total {len(mp4_files)} video(s)."
    )


if __name__ == "__main__":
    main()
