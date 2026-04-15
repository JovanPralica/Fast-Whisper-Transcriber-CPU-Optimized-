Fast Whisper Transcriber (CPU Optimized)
Overview

This project is a high-performance transcription and subtitle generator built on top of:

faster-whisper (CTranslate2 backend)
OpenAI Whisper model architecture

It is designed for:

Large audio files
Batch processing (multiple files / folders)
CPU-only environments
Fast and stable transcription
What is Whisper?

Whisper is a general-purpose speech recognition model trained on a large and diverse dataset of audio.

It supports:

Multilingual speech recognition
Speech translation
Language identification
Voice activity detection
Model architecture

Whisper uses a Transformer sequence-to-sequence model.

Instead of separate pipelines, it:

Encodes audio → tokens
Uses a decoder to predict:
text
language
task (transcribe / translate)

All tasks are handled in a single unified model using special tokens.

Why this implementation is faster

This project does not use standard openai-whisper for inference.

Instead, it uses:

faster-whisper
Built on CTranslate2
Optimized inference engine
Supports quantization
Key optimizations
INT8 quantization → major CPU speedup
Chunked processing → avoids memory spikes
Model loaded once → reused across files
Forced language option → skips detection step
Sequential chunking → stable for large files
Result

Compared to standard Whisper:

~2x–4x faster on CPU
Lower memory usage
More stable on large files
Why Anaconda is the best way to run this

Using Anaconda (or Miniconda) provides:

Isolated environment → no dependency conflicts
Controlled Python version
Stable package management
Clean system (no pollution)
Recommended setup
conda create -n whisper_env python=3.10
conda activate whisper_env
pip install faster-whisper

This avoids:

broken dependencies
PATH issues
PyTorch / tokenizer conflicts
Requirements
Python
3.8 – 3.11 supported
FFmpeg (REQUIRED)
Windows (recommended)
choco install ffmpeg

or

scoop install ffmpeg

Verify installation:

ffmpeg -version
ffprobe -version
Supported Models
Model	Speed	Accuracy
tiny	fastest	lowest
base	fast	good
small	medium	very good
medium	slow	high
large	slowest	highest
Recommended for CPU
-m base --compute-type int8 --beam-size 1
Input Support

You can pass:

Single file
python fast_transcribe.py file.mp3
Multiple files
python fast_transcribe.py file1.mp3 file2.mp3
Entire folder (recommended)
python fast_transcribe.py "C:\AudioFolder"

Supported formats:

.mp3
.wav
.m4a
.mp4
Output Modes
1. Transcript Mode

Output: .txt

Options:

Plain text
Optional timestamps
2. Subtitle Mode

Output: .srt

Example:

1
00:00:01,000 --> 00:00:03,000
Text here

Compatible with:

VLC
YouTube
Premiere Pro
CapCut
Translation

Whisper supports:

Same-language transcription
Translation → English only

Examples:

Russian → English subtitles (supported)
Serbian → English subtitles (supported)
Serbian → German (not supported without external tools)
How it works
Convert audio → 16kHz mono WAV
Split into chunks (default: 300 seconds)
Transcribe each chunk
Reconstruct full output
Save as:
.txt (transcript)
.srt (subtitles)
Usage
Recommended command
python fast_transcribe.py "C:\YourFolder" -m base --compute-type int8 --beam-size 1 --language sr --chunk-seconds 300
Parameters
Argument	Description
-m	Model size
--compute-type	Use int8 for CPU performance
--beam-size	1 = fastest decoding
--language	Force language (recommended)
--chunk-seconds	Chunk size for large files
Language selection

Examples:

--language sr   # Serbian
--language en   # English
--language ru   # Russian

Recommendation:
Always specify language for better accuracy and speed.

Official Whisper vs This Setup
openai-whisper (standard)
Simpler usage
Slower inference
Loads full audio
Higher RAM usage
This project (optimized)
Faster inference engine
Chunked processing
Lower memory usage
Better for long files
More control over performance
Summary

This setup is optimized for:

CPU performance
Large audio files
Batch workflows
Stability

If standard Whisper is slow or crashes on large files, this approach solves those issues.
