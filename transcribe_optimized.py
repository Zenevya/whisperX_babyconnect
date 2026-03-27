#!/usr/bin/env python3
"""
Minimal WhisperX transcription pipeline.

Purpose:
- Keep the script as simple as possible.
- Transcribe full files directly.
- Preserve CSV output format expected by run_pipeline.py.
"""

import os
import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch

try:
    import whisperx
except Exception:
    whisperx = None


def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def load_model(model_name: str = "medium", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if whisperx is not None:
        try:
            print(f"Loading WhisperX model '{model_name}' on {device}")
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type="float32",
                language="en",
            )
            model._is_whisperx = True
            return model
        except Exception as e:
            print(f"WhisperX load failed: {e}")

    try:
        import whisper
        print(f"Falling back to local OpenAI Whisper '{model_name}' on {device}")
        model = whisper.load_model(model_name, device=device)
        model._is_whisperx = False
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load either WhisperX or Whisper: {e}")


def transcribe_file(audio_path: str, model):
    if getattr(model, "_is_whisperx", False):
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=1)
    else:
        result = model.transcribe(
            audio_path,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
        )

    rows = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text", "") or "").strip()

        if not text:
            continue

        rows.append({
            "Chunk": i,
            "Chunk Start": format_timestamp(start),
            "Chunk End": format_timestamp(end),
            "Start Time": format_timestamp(start),
            "End Time": format_timestamp(end),
            "Start Seconds": start,
            "End Seconds": end,
            "Transcript": text,
            "flags": "",
            "hallucination_score": 0.0,
        })

    return rows


def process_directory(input_dir: str, output_dir: str, model_name: str = "medium"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_name=model_name, device=device)

    for file in sorted(input_path.iterdir()):
        if file.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac"}:
            continue

        try:
            print(f"\nProcessing file: {file.name}")
            rows = transcribe_file(str(file), model)
            df = pd.DataFrame(rows)

            if not df.empty and "Start Seconds" in df.columns:
                df = df.sort_values("Start Seconds")

            if "Start Seconds" in df.columns:
                df = df.drop(columns=["Start Seconds"])
            if "End Seconds" in df.columns:
                df = df.drop(columns=["End Seconds"])

            out_csv = output_path / f"transcriptions_{file.stem}_optimized.csv"
            df.to_csv(out_csv, index=False)
            print(f"Saved transcript: {out_csv}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="medium")
    return parser.parse_args()


def main():
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, model_name=args.model)


if __name__ == "__main__":
    main()