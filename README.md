## Optimized WhisperX Pipeline

A modular speech-to-text pipeline built on WhisperX, designed for robust transcription, preprocessing experimentation, and WER evaluation.

This project focuses on improving transcription quality for real-world audio, such as home recordings and child speech, through techniques including:
- noise reduction
- chunk-based transcription
- voice activity detection (VAD)

## Project Structure

Optimized-Whisper_X-Pipeline/
├── 1_incoming_files/        # Raw input files placed here for processing
├── 2_processing_audio/      # Temporary processing folder used during pipeline execution
├── 3_completed_runs/        # Final organized outputs for each processed audio file
│
├── preprocessing/           # Modular preprocessing components
│   ├── vad.py               # Voice activity detection logic
│   ├── chunking.py          # Audio chunk splitting and merging
│   └── noise_reduction.py   # Audio enhancement and normalization
│
├── scripts/
│   ├── run_pipeline.py          # Main pipeline entry point
│   ├── transcribe_optimized.py  # Transcription engine
│   ├── calculate_wer.py         # Standalone WER calculation script
│   └── clean_cha.py             # Utility for converting .cha transcripts to clean text
│
└── requirements.txt

## How It Works

### Pipeline Overview

1_incoming_files/
    ↓
run_pipeline.py
    ↓
2_processing_audio/
    ↓
transcribe_optimized.py
    ↓
calculate_wer.py
    ↓
3_completed_runs/<audio_id>/

Each audio file is processed independently and produces the following output structure:

3_completed_runs/<audio_id>/
├── audio/          # Original audio file
├── reference/      # Ground-truth transcript, if available
├── transcript/     # Generated transcript output (CSV)
└── metrics/        # WER results or error logs

## Features

### Chunk-Based Transcription
- Splits audio into manageable segments
- Helps reduce long-context hallucinations
- Makes future batching and parallel processing easier

### Preprocessing
- Noise reduction
- Volume normalization
- Dynamic range compression

### Voice Activity Detection (VAD)
- Uses VAD to identify active speech regions before transcription
- Helps reduce silence-heavy input
- Improves chunking efficiency
- Handles no-speech cases gracefully

### WER Evaluation
- Automatically compares generated transcripts against reference transcripts
- WER logic is separated into its own script for easier debugging and reuse
- Outputs results to:

`3_completed_runs/<audio_id>/metrics/wer.txt`

## Configuration Options

Inside `run_pipeline.py`, you can adjust pipeline behavior with options such as:
- `--use-vad` — enable VAD-based segmentation
- `--enhance` — enable preprocessing and audio enhancement
- `--model` — choose the Whisper model size (e.g. `small`, `medium`, `large`)