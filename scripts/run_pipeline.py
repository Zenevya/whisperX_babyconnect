#!/usr/bin/env python3

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from jiwer import wer


ROOT = Path(__file__).resolve().parent.parent

INCOMING = ROOT / "1_incoming_files"
PROCESSING = ROOT / "2_processing_audio"
COMPLETED = ROOT / "3_completed_runs"
TRANSCRIBE_SCRIPT = ROOT / "scripts" / "transcribe_optimized.py"
WER_SCRIPT = ROOT / "scripts" / "calculate_wer.py"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

def ensure_dirs() -> None:
    for folder in [INCOMING, PROCESSING, COMPLETED]:
        folder.mkdir(parents=True, exist_ok=True)

def run_wer(reference_txt: Path, transcript_csv: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(WER_SCRIPT),
        "--ref",
        str(reference_txt),
        "--hyp",
        str(transcript_csv),
        "--out",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

def find_reference_file(audio_stem: str) -> Path | None:
    candidate = INCOMING / f"clean_{audio_stem}.txt"
    return candidate if candidate.exists() else None

def run_whisperx(audio_path: Path, output_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(TRANSCRIBE_SCRIPT),
        "--audio",
        str(audio_path),
        "--output-csv",
        str(output_csv),
        "--model",
        "medium",
        "--enhance",
    ]
    subprocess.run(cmd, check=True)


def process_one_audio(audio_path: Path) -> None:
    print(f"\nProcessing: {audio_path.name}")

    audio_stem = audio_path.stem
    reference_file = find_reference_file(audio_stem)

    processing_path = PROCESSING / audio_path.name
    shutil.move(str(audio_path), str(processing_path))

    run_folder = COMPLETED / audio_stem
    audio_dir = run_folder / "audio"
    ref_dir = run_folder / "reference"
    transcript_dir = run_folder / "transcript"
    metrics_dir = run_folder / "metrics"

    for folder in [audio_dir, ref_dir, transcript_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    final_audio_path = audio_dir / processing_path.name
    shutil.move(str(processing_path), str(final_audio_path))

    final_reference_path = None
    if reference_file and reference_file.exists():
        final_reference_path = ref_dir / reference_file.name
        shutil.move(str(reference_file), str(final_reference_path))

    final_transcript_path = transcript_dir / f"transcriptions_{audio_stem}_optimized.csv"
    
    if not final_audio_path.exists():
        raise FileNotFoundError(f"Audio not found before transcription: {final_audio_path}")

    run_whisperx(final_audio_path, final_transcript_path)

    if not final_transcript_path.exists():
        raise FileNotFoundError(f"Transcript CSV was not created: {final_transcript_path}")

    if final_reference_path and final_reference_path.exists():
        try:
            wer_file = metrics_dir / "wer.txt"
            run_wer(final_reference_path, final_transcript_path, wer_file)
            # Read and display only the WER line
            if wer_file.exists():
                wer_lines = [line.strip() for line in wer_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                if wer_lines:
                    print(f"WER for {audio_stem}: {wer_lines[0]}")
            print(f"WER calculated for {audio_stem}")
        except Exception as e:
            error_file = metrics_dir / "wer_error.txt"
            error_file.write_text(str(e), encoding="utf-8")
            print(f"WER computation failed for {audio_stem}: {e}")
    else:
        note_file = metrics_dir / "wer.txt"
        note_file.write_text("No matching ground-truth transcript found.\n", encoding="utf-8")
        print(f"No matching ground-truth transcript found for {audio_stem}")

def main() -> None:
    ensure_dirs()

    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(INCOMING.glob(f"*{ext}"))

    if not audio_files:
        print("No new audio files found in 1_incoming_files.")
        return

    for audio_file in sorted(audio_files):
        try:
            process_one_audio(audio_file)
        except Exception as e:
            print(f"Failed on {audio_file.name}: {e}")


if __name__ == "__main__":
    main()