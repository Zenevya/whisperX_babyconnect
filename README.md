## Optimized WhisperX Pipeline
A modular speech-to-text pipeline built on Whisper/WhisperX, designed for robust transcription, preprocessing experimentation, and WER evaluation.

This project focuses on improving transcription quality for real-world audio (e.g., home recordings, child speech) through preprocessing techniques such as:
- noise reduction
- chunk-based transcription
- voice activity detection (VAD)

## Project Structure
Optimized-Whisper_X-Pipeline/
│
├── 1_incoming_files/        # Raw input files (audio + ground truth)
├── 2_processing_audio/      # Temporary processing folder
├── 3_completed_runs/        # Final outputs organized per audio file
│
├── preprocessing/           # Modular preprocessing logic
│   ├── vad.py               # Voice Activity Detection (energy-based)
│   ├── chunking.py          # Chunk splitting and merging
│   └── noise_reduction.py   # Audio cleaning & normalization
│
├── scripts/
│   ├── run_pipeline.py      # Main pipeline (entry point)
│   └── transcribe_optimized.py  # Transcription engine
│
└── requirements.txt

## How It Works
Pipeline Overview
1_incoming_files/
    ↓
run_pipeline.py
    ↓
2_processing_audio/
    ↓
transcribe_optimized.py
    ↓
3_completed_runs/<audio_id>/

Each audio file is processed independently and produces:

3_completed_runs/<audio_id>/
├── audio/          # Original audio
├── reference/      # Ground truth transcript
├── transcript/     # Generated transcript (CSV)
└── metrics/        # WER evaluation

## Features
### Chunk-Based Transcription
Audio is split into ~30 second segments
Prevents long-context hallucinations
Enables future batching and parallelism

### Preprocessing (Modular)
Noise reduction
Volume normalization
Dynamic range compression

### Voice Activity Detection (VAD)
Lightweight energy-based VAD
Removes silence and improves chunking
Falls back gracefully if no speech detected

### WER Evaluation
Automatically compares transcript with ground truth
Outputs results in:
3_completed_runs/<audio_id>/metrics/wer.txt

## Configuration Options
Inside run_pipeline.py, you can adjust:
"--use-vad"      # Enable VAD segmentation
"--enhance"      # Enable noise reduction + normalization
"--model"        # Whisper model (small, medium, large)
