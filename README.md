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

Instead of separate pipelines (ASR, translation, detection), it:

Encodes audio → tokens
Uses a decoder to predict:
text
language
task (transcribe / translate)

All tasks are handled in a single unified model using special tokens.

Why this implementation is faster

This project does NOT use standard openai-whisper for inference.

Instead, it uses:

faster-whisper
Built on CTranslate2
Optimized inference engine
Supports quantization
Key optimizations
INT8 quantization → major CPU speedup
Chunked processing → avoids memory spikes
Model loaded once → reused across files
Optional language forcing → skips detection step
Sequential chunk processing → stable for large files
Result

Compared to standard Whisper:

~2x to 4x faster on CPU
Lower memory usage
More stable on large files
Why Anaconda is the best way to run this

Using Anaconda (or Miniconda) gives:

Clean isolated environment (no dependency conflicts)
Easy Python version control
Reliable package management
No system pollution
Recommended setup
conda create -n whisper_env python=3.10
conda activate whisper_env
pip install faster-whisper

This avoids:

broken dependencies
PATH issues
version conflicts with PyTorch / tokenizers
Requirements
Python
3.8 – 3.11 supported
FFmpeg (required)

Install depending on OS:

Windows (recommended):

choco install ffmpeg

or

scoop install ffmpeg

Verify:

ffmpeg -version
ffprobe -version
Supported Models
Model	Speed	Accuracy
tiny	fastest	lowest
base	fast	good
small	medium	very good
medium	slow	high
large	slowest	highest
Recommended (CPU)
-m base --compute-type int8 --beam-size 1
Input Support

You can pass:

Single file
python fast_transcribe.py file.mp3
Multiple files
python fast_transcribe.py file1.mp3 file2.mp3
Entire folder
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

plain text
optional timestamps
2. Subtitle Mode

Output: .srt

Format:

1
00:00:01,000 --> 00:00:03,000
Text here

Compatible with:

VLC
YouTube
Premiere
CapCut
Translation

Whisper supports:

Same-language transcription
Translation → English only

Example:

Russian → English subtitles ✅
Serbian → English subtitles ✅
Serbian → German ❌ (requires external model)
How it works
Converts audio → 16kHz mono WAV
Splits into chunks (default: 300 seconds)
Processes each chunk with Whisper
Reconstructs full output
Saves:
.txt or .srt
Usage
Recommended command
python fast_transcribe.py "C:\YourFolder" -m base --compute-type int8 --beam-size 1 --language sr --chunk-seconds 300
Parameters
Argument	Description
-m	Model size
--compute-type	Use int8 for CPU
--beam-size	1 = fastest
--language	Force language (recommended)
--chunk-seconds	Split size
Language selection

Examples:

--language sr   # Serbian
--language en   # English
--language ru   # Russian

For Balkan languages, always specify manually for best results.

Notes on official Whisper vs this setup
Official (openai-whisper)
Simpler
Slower
Loads full audio
Higher RAM usage
This project
Faster inference engine
Chunked processing
Better for long audio
More control over performance
Summary

This setup is optimized for:

CPU performance
Large audio files
Batch workflows
Stability

If standard Whisper is slow or crashes on large files, this approach solves those issues.

If you want next upgrade:

parallel processing (multi-core CPU)
translation to any language (not just English)

I can extend the script further.
