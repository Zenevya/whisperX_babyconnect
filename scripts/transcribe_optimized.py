#!/usr/bin/env python3
"""
Chunk-based WhisperX transcription.

Intended location:
    scripts/transcribe_optimized.py

Purpose:
- Transcribe a single audio file at a time.
- Write the transcript directly to its final CSV path.
- Keep preprocessing logic in separate modules:
    preprocessing/vad.py
    preprocessing/chunking.py
    preprocessing/noise_reduction.py
- Preserve CSV output format expected by the pipeline.
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch

# Make project root importable when this file lives in /scripts
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import whisperx
except Exception:
    whisperx = None

try:
    import whisper
except Exception:
    whisper = None

from preprocessing.chunking import build_fixed_chunks, slice_audio
from preprocessing.noise_reduction import (
    reduce_noise,
    normalize_audio,
    compress_audio,
)
from preprocessing.vad import get_vad_segments


SAMPLE_RATE = 16000
DEFAULT_MODEL = "medium"
DEFAULT_CHUNK_SIZE = 30.0


def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    device = device or get_device()

    if whisperx is not None:
        try:
            print(f"Loading WhisperX model '{model_name}' on {device}")
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type="float32",
                language="en",
                asr_options={
                    "condition_on_previous_text": False,
                    "temperatures": [0.0],
                    "beam_size": 5,
                    "no_speech_threshold": 0.6,
                    "compression_ratio_threshold": 2.4,
                },
            )
            model._is_whisperx = True
            return model
        except Exception as e:
            print(f"WhisperX load failed: {e}")

    if whisper is not None:
        try:
            print(f"Falling back to OpenAI Whisper '{model_name}' on {device}")
            model = whisper.load_model(model_name, device=device)
            model._is_whisperx = False
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load WhisperX or Whisper: {e}")

    raise RuntimeError("Neither whisperx nor whisper could be imported.")


def load_audio_file(audio_path: str):
    if whisperx is None:
        raise RuntimeError("whisperx is required for audio loading in this script.")
    return whisperx.load_audio(audio_path)


def get_audio_duration(audio_array, sample_rate: int = SAMPLE_RATE) -> float:
    return len(audio_array) / float(sample_rate)


def apply_preprocessing(audio, sample_rate: int = SAMPLE_RATE, enhance: bool = False):
    """
    Keep preprocessing optional and simple.
    """
    if not enhance:
        return audio

    audio = reduce_noise(audio, sample_rate=sample_rate)
    audio = normalize_audio(audio)
    audio = compress_audio(audio)
    return audio


def build_chunks(audio, sample_rate: int, chunk_size: float, use_vad: bool):
    """
    Returns a list of chunks:
        [{"start": float, "end": float}, ...]
    """
    duration = get_audio_duration(audio, sample_rate=sample_rate)

    if use_vad:
        try:
            vad_segments = get_vad_segments(audio, sample_rate=sample_rate)
            if vad_segments:
                return vad_segments
            print("VAD returned no segments. Falling back to fixed chunking.")
        except Exception as e:
            print(f"VAD failed ({e}). Falling back to fixed chunking.")

    return build_fixed_chunks(duration, chunk_size=chunk_size)


def transcribe_chunk_with_whisperx(chunk_audio, model):
    return model.transcribe(
        chunk_audio,
        batch_size=1,
        language="en",
    )


def transcribe_chunk_with_whisper(chunk_audio, model):
    return model.transcribe(
        chunk_audio,
        language="en",
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
    )


def transcribe_file(
    audio_path: str,
    model,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
    use_vad: bool = True,
    enhance: bool = False,
):
    audio = load_audio_file(audio_path)
    audio = apply_preprocessing(audio, sample_rate=SAMPLE_RATE, enhance=enhance)

    chunks = build_chunks(
        audio,
        sample_rate=SAMPLE_RATE,
        chunk_size=chunk_size,
        use_vad=use_vad,
    )

    rows = []
    chunk_id = 1

    for chunk in chunks:
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])
        chunk_audio = slice_audio(audio, chunk_start, chunk_end, SAMPLE_RATE)

        if len(chunk_audio) == 0:
            continue

        if getattr(model, "_is_whisperx", False):
            result = transcribe_chunk_with_whisperx(chunk_audio, model)
        else:
            result = transcribe_chunk_with_whisper(chunk_audio, model)

        for seg in result.get("segments", []):
            rel_start = float(seg.get("start", 0.0))
            rel_end = float(seg.get("end", 0.0))
            text = (seg.get("text", "") or "").strip()

            if not text:
                continue

            abs_start = chunk_start + rel_start
            abs_end = chunk_start + rel_end

            rows.append(
                {
                    "Chunk": chunk_id,
                    "Chunk Start": format_timestamp(chunk_start),
                    "Chunk End": format_timestamp(chunk_end),
                    "Start Time": format_timestamp(abs_start),
                    "End Time": format_timestamp(abs_end),
                    "Start Seconds": abs_start,
                    "End Seconds": abs_end,
                    "Transcript": text,
                    "flags": "",
                    "hallucination_score": 0.0,
                }
            )

        chunk_id += 1

    if not rows:
        print("WARNING: No transcription segments produced.")

    return rows


def write_transcript_csv(rows, output_csv: str):
    df = pd.DataFrame(rows)

    if not df.empty and "Start Seconds" in df.columns:
        df = df.sort_values("Start Seconds")

    if "Start Seconds" in df.columns:
        df = df.drop(columns=["Start Seconds"])
    if "End Seconds" in df.columns:
        df = df.drop(columns=["End Seconds"])

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved transcript: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True, help="Path to a single audio file")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")

    parser.add_argument("--model", default="medium")
    parser.add_argument("--chunk-size", type=float, default=30.0)

    parser.add_argument("--use-vad", action="store_true")
    parser.add_argument("--enhance", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_name=args.model, device=device)

    rows = transcribe_file(
        audio_path=args.audio,
        model=model,
        chunk_size=args.chunk_size,
        use_vad=args.use_vad,
        enhance=args.enhance,
    )

    write_transcript_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()