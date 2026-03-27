#!/usr/bin/env python3

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from jiwer import wer


ROOT = Path(__file__).resolve().parent
INCOMING = ROOT / "incoming_audio"
PROCESSING = ROOT / "processing_audio"
OUTPUT = ROOT / "output_transcripts"
COMPLETED = ROOT / "completed_runs"
REF_TRANSCRIPTS = ROOT / "ref_transcripts"

TRANSCRIBE_SCRIPT = ROOT / "transcribe_optimized.py"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dirs() -> None:
    for folder in [INCOMING, PROCESSING, OUTPUT, COMPLETED]:
        folder.mkdir(exist_ok=True)


def find_transcript_file(audio_stem: str) -> Path | None:
    candidates = list(OUTPUT.glob(f"*{audio_stem}*.csv"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def find_reference_file(audio_stem: str) -> Path | None:
    candidate = REF_TRANSCRIPTS / f"clean_{audio_stem}.txt"
    return candidate if candidate.exists() else None


def compute_wer_from_files(reference_txt: Path, transcript_csv: Path) -> float:
    reference_text = reference_txt.read_text(encoding="utf-8", errors="ignore")
    df = pd.read_csv(transcript_csv)

    if "Transcript" not in df.columns:
        raise ValueError(f"'Transcript' column not found in {transcript_csv.name}")

    hypothesis_text = " ".join(df["Transcript"].dropna().astype(str).tolist())

    reference_text = normalize_text(reference_text)
    hypothesis_text = normalize_text(hypothesis_text)

    return wer(reference_text, hypothesis_text)


def run_whisperx() -> None:
    cmd = [
        "python",
        str(TRANSCRIBE_SCRIPT),
        "--input-dir",
        str(PROCESSING),
        "--output-dir",
        str(OUTPUT),
    ]
    subprocess.run(cmd, check=True)


def process_one_audio(audio_path: Path) -> None:
    print(f"\nProcessing: {audio_path.name}")

    processing_path = PROCESSING / audio_path.name
    shutil.move(str(audio_path), str(processing_path))

    run_whisperx()

    audio_stem = processing_path.stem
    transcript_file = find_transcript_file(audio_stem)

    run_folder = COMPLETED / processing_path.stem
    audio_dir = run_folder / "audio"
    whisper_dir = run_folder / "whisperx"
    ref_dir = run_folder / "reference"
    metrics_dir = run_folder / "metrics"

    for folder in [audio_dir, whisper_dir, ref_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    final_audio_path = audio_dir / processing_path.name
    shutil.move(str(processing_path), str(final_audio_path))

    if transcript_file and transcript_file.exists():
        final_transcript_path = whisper_dir / transcript_file.name
        shutil.copy2(transcript_file, final_transcript_path)
    else:
        print(f"No transcript CSV found for {audio_stem}")
        return

    reference_file = find_reference_file(audio_stem)
    if reference_file and reference_file.exists():
        final_reference_path = ref_dir / reference_file.name
        shutil.copy2(reference_file, final_reference_path)

        try:
            score = compute_wer_from_files(final_reference_path, final_transcript_path)
            wer_file = metrics_dir / "wer.txt"
            wer_file.write_text(f"WER: {score:.4f}\n", encoding="utf-8")
            print(f"WER for {audio_stem}: {score:.4f}")
        except Exception as e:
            error_file = metrics_dir / "wer_error.txt"
            error_file.write_text(str(e), encoding="utf-8")
            print(f"WER computation failed for {audio_stem}: {e}")
    else:
        note_file = metrics_dir / "wer.txt"
        note_file.write_text("No matching cleaned reference transcript found.\n", encoding="utf-8")
        print(f"No matching cleaned reference found for {audio_stem}")


def main() -> None:
    ensure_dirs()

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        audio_files.extend(INCOMING.glob(ext))

    if not audio_files:
        print("No new audio files found in incoming_audio.")
        return

    for audio_file in sorted(audio_files):
        try:
            process_one_audio(audio_file)
        except Exception as e:
            print(f"Failed on {audio_file.name}: {e}")


if __name__ == "__main__":
    main()