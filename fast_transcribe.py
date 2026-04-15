import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from faster_whisper import WhisperModel


def run_ffmpeg_command(cmd):
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("[FFMPEG ERROR]")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print("[ERROR] ffmpeg is not installed or not in PATH.")
        raise


def get_audio_duration(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def convert_to_wav(input_path: str, output_wav: str):
    print("[INFO] Converting input to 16kHz mono WAV...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_wav
    ]
    run_ffmpeg_command(cmd)
    print(f"[INFO] WAV created: {output_wav}")


def split_audio(input_wav: str, chunk_seconds: int, temp_dir: str) -> list[str]:
    duration = get_audio_duration(input_wav)
    if duration <= 0:
        raise RuntimeError("Could not determine audio duration.")

    total_chunks = math.ceil(duration / chunk_seconds)
    print(f"[INFO] Audio duration: {duration:.2f} sec")
    print(f"[INFO] Splitting into {total_chunks} chunk(s) of {chunk_seconds} sec")

    chunk_paths = []
    for i in range(total_chunks):
        start = i * chunk_seconds
        out_path = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_wav,
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-ac", "1",
            "-ar", "16000",
            out_path
        ]
        run_ffmpeg_command(cmd)
        chunk_paths.append(out_path)
        print(f"[INFO] Created chunk {i + 1}/{total_chunks}")

    return chunk_paths


def format_txt_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcribe_chunks(
    model,
    chunk_paths: list[str],
    output_path: str,
    beam_size: int,
    language: str | None,
    vad_filter: bool,
    use_timestamps: bool,
    chunk_seconds: int,
    mode: str,
    task: str,
):
    overall_start = time.time()
    total_chunks = len(chunk_paths)
    subtitle_index = 1

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunk_paths, start=1):
            print("=" * 60)
            print(f"[INFO] Transcribing chunk {idx}/{total_chunks}")
            print(f"[INFO] File: {chunk}")
            chunk_start = time.time()

            segments, info = model.transcribe(
                chunk,
                beam_size=beam_size,
                language=language,
                vad_filter=vad_filter,
                task=task,
            )

            wrote_anything = False
            chunk_offset = (idx - 1) * chunk_seconds

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                real_start = segment.start + chunk_offset
                real_end = segment.end + chunk_offset

                if mode == "transcript":
                    if use_timestamps:
                        start_ts = format_txt_timestamp(real_start)
                        end_ts = format_txt_timestamp(real_end)
                        f.write(f"[{start_ts} -> {end_ts}] {text}\n")
                    else:
                        f.write(text + "\n")

                elif mode == "subtitle":
                    start_ts = format_srt_timestamp(real_start)
                    end_ts = format_srt_timestamp(real_end)
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_ts} --> {end_ts}\n")
                    f.write(f"{text}\n\n")
                    subtitle_index += 1

                wrote_anything = True

            if idx == 1:
                detected = getattr(info, "language", "unknown")
                prob = getattr(info, "language_probability", None)
                print(f"[INFO] Detected language: {detected}")
                if prob is not None:
                    print(f"[INFO] Language confidence: {prob:.4f}")

            elapsed = time.time() - chunk_start
            print(f"[INFO] Chunk {idx}/{total_chunks} finished in {elapsed:.2f} sec")
            if not wrote_anything:
                print("[INFO] No text detected in this chunk.")

    total_elapsed = time.time() - overall_start
    print("=" * 60)
    print(f"[DONE] Output saved to: {os.path.abspath(output_path)}")
    print(f"[DONE] Total transcription time: {total_elapsed:.2f} sec")
    print("=" * 60)


def process_file(
    model,
    input_path: str,
    output_path: str,
    beam_size: int,
    language: str | None,
    vad_filter: bool,
    chunk_seconds: int,
    use_timestamps: bool,
    mode: str,
    task: str,
):
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return

    print("\n" + "#" * 70)
    print(f"[PROCESSING] {input_path}")
    print(f"[OUTPUT]     {output_path}")
    print(f"[MODE]       {mode}")
    print(f"[TASK]       {task}")
    print("#" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(temp_dir, "normalized.wav")
        convert_to_wav(input_path, wav_path)
        chunk_paths = split_audio(wav_path, chunk_seconds, temp_dir)
        transcribe_chunks(
            model=model,
            chunk_paths=chunk_paths,
            output_path=output_path,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            use_timestamps=use_timestamps,
            chunk_seconds=chunk_seconds,
            mode=mode,
            task=task,
        )


def collect_files(inputs):
    supported = (".mp3", ".wav", ".m4a", ".mp4")
    files = []

    for item in inputs:
        item = os.path.abspath(item)

        if os.path.isfile(item):
            if item.lower().endswith(supported):
                files.append(item)
            else:
                print(f"[WARNING] Skipped unsupported file: {item}")

        elif os.path.isdir(item):
            for f in os.listdir(item):
                full_path = os.path.join(item, f)
                if os.path.isfile(full_path) and full_path.lower().endswith(supported):
                    files.append(full_path)

        else:
            print(f"[WARNING] Skipped (not found): {item}")

    return sorted(files)


def choose_mode():
    while True:
        print("Choose output mode:")
        print("1 = Transcript (.txt)")
        print("2 = Subtitle (.srt)")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return "transcript"
        if choice == "2":
            return "subtitle"

        print("[ERROR] Invalid choice. Try again.\n")


def choose_transcript_timestamps():
    while True:
        choice = input("Use timestamps in transcript? (Y/N): ").strip().lower()
        if choice in ("y", "n"):
            return choice == "y"
        print("[ERROR] Enter Y or N.")


def choose_subtitle_task():
    while True:
        print("\nSubtitle mode:")
        print("1 = Same-language subtitles")
        print("2 = Translate subtitles to English")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return "transcribe"
        if choice == "2":
            return "translate"

        print("[ERROR] Invalid choice. Try again.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fast chunked transcription/subtitle generator using faster-whisper for files or folders."
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="One or more input files or folders"
    )

    parser.add_argument(
        "-m",
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run on"
    )

    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "int8_float32", "float32", "float16"],
        help="Computation type"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size (1 is fastest)"
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Source language code like en, sr, hr, ru, de. Leave empty for auto-detect."
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD silence filtering"
    )

    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=300,
        help="Chunk size in seconds for large files"
    )

    args = parser.parse_args()

    mode = choose_mode()

    use_timestamps = False
    task = "transcribe"

    if mode == "transcript":
        use_timestamps = choose_transcript_timestamps()

    elif mode == "subtitle":
        task = choose_subtitle_task()

    files = collect_files(args.input)

    if not files:
        print("[ERROR] No valid audio/video files found.")
        sys.exit(1)

    print("=" * 70)
    print("FAST CHUNKED MULTI-FILE / FOLDER TRANSCRIBE")
    print("=" * 70)
    print(f"Files found    : {len(files)}")
    print(f"Model          : {args.model}")
    print(f"Device         : {args.device}")
    print(f"Compute type   : {args.compute_type}")
    print(f"Beam size      : {args.beam_size}")
    print(f"Source language: {args.language if args.language else 'auto'}")
    print(f"VAD filter     : {not args.no_vad}")
    print(f"Chunk seconds  : {args.chunk_seconds}")
    print(f"Mode           : {mode}")
    print(f"Task           : {task}")
    if mode == "transcript":
        print(f"Timestamps     : {use_timestamps}")
    print("=" * 70)

    print("[INFO] Loading model once...")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print("[INFO] Model loaded.\n")

    for input_path in files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        if mode == "transcript":
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}.txt")
        else:
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}.srt")

        try:
            process_file(
                model=model,
                input_path=input_path,
                output_path=output_path,
                beam_size=args.beam_size,
                language=args.language,
                vad_filter=not args.no_vad,
                chunk_seconds=args.chunk_seconds,
                use_timestamps=use_timestamps,
                mode=mode,
                task=task,
            )
        except Exception as e:
            print(f"[ERROR] Failed on file: {input_path}")
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()