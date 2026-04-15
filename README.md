Fast Whisper Transcriber

[Blog] [Paper] [Model card] [Colab example]

Fast Whisper Transcriber is a high-performance speech recognition and subtitle generation tool built on top of Whisper. It is optimized for CPU usage, large audio files, and batch processing, using the faster-whisper backend.

It supports:

multilingual transcription
subtitle generation (.srt)
translation to English
processing multiple files or entire folders

Compared to the standard Whisper implementation, this setup is designed to be significantly faster and more stable, especially on CPU-only systems.

Approach

A Transformer sequence-to-sequence model (Whisper) is used for speech processing tasks including:

multilingual speech recognition
speech translation
language identification
voice activity detection

These tasks are represented as a sequence of tokens predicted by the decoder, allowing a single model to replace traditional multi-stage pipelines.

This project uses:

faster-whisper (CTranslate2 backend) for optimized inference
INT8 quantization for CPU acceleration
chunked processing for large audio files

Instead of processing the entire file at once, audio is:

converted to 16kHz mono
split into chunks (default: 300 seconds)
processed sequentially
reconstructed into final output

This avoids memory issues and improves performance on long recordings.

Setup

The original Whisper setup uses:

Python 3.8–3.11
PyTorch
tiktoken

However, this project is designed to run more efficiently using Anaconda + faster-whisper.

Recommended (best setup)
conda create -n whisper_env python=3.10
conda activate whisper_env
pip install faster-whisper

This is the recommended way to run the project, because it provides:

isolated environment (no dependency conflicts)
stable package versions
cleaner installation process
better compatibility with CTranslate2
FFmpeg (required)

Install ffmpeg:

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# MacOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Windows (Scoop)
scoop install ffmpeg

Verify installation:

ffmpeg -version
ffprobe -version
Available models and performance

Whisper provides multiple model sizes with speed/accuracy tradeoffs:

Size	Parameters	Speed	Accuracy
tiny	39M	fastest	lowest
base	74M	fast	good
small	244M	medium	very good
medium	769M	slow	high
large	1550M	slowest	highest
Recommended for CPU
-m base --compute-type int8 --beam-size 1

This provides the best balance of:

speed
accuracy
memory usage
Command-line usage

Process a single file:

python fast_transcribe.py audio.mp3

Process multiple files:

python fast_transcribe.py file1.mp3 file2.mp3

Process an entire folder:

python fast_transcribe.py "C:\YourFolder"
Recommended command (optimized)
python fast_transcribe.py "C:\YourFolder" -m base --compute-type int8 --beam-size 1 --language sr --chunk-seconds 300
Output modes
Transcript mode

Outputs .txt files:

plain text
optional timestamps
Subtitle mode

Outputs .srt files:

1
00:00:01,000 --> 00:00:03,000
Text here

Compatible with:

VLC
YouTube
Premiere Pro
CapCut
Translation

Supports:

same-language transcription
translation → English

Examples:

Russian → English subtitles (supported)
Serbian → English subtitles (supported)
Serbian → German (not supported without external tools)
Language selection

You can manually set language:

--language sr   # Serbian
--language en   # English
--language ru   # Russian

Specifying language:

improves accuracy
reduces startup time
Python usage

Example using Whisper directly:

import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])

This project uses an optimized pipeline instead of processing full audio at once.

How this differs from standard Whisper
Standard Whisper
slower on CPU
processes entire file
higher memory usage
This project
faster inference (faster-whisper)
chunked processing
lower memory usage
better for large files
batch processing support
Summary

This project is optimized for:

CPU-based systems
large audio files
multiple file workflows
stable transcription

It is the recommended approach if standard Whisper is too slow or unstable on your machine.
