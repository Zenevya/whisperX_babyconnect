#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
from jiwer import wer


def normalize_text(text: str) -> str:
    text = text.lower()
    # remove unwanted punctuation and non-word characters
    text = re.sub(r"[^\w\s']", " ", text)
    # remove explicit digit 0 tokens (and optionally any standalone 0s)
    text = re.sub(r"\b0\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_from_files(reference_txt: Path, transcript_csv: Path):
    reference_text = reference_txt.read_text(encoding="utf-8", errors="ignore")
    df = pd.read_csv(transcript_csv)

    if "Transcript" not in df.columns:
        raise ValueError(f"'Transcript' column not found in {transcript_csv.name}")

    hypothesis_text = " ".join(df["Transcript"].dropna().astype(str).tolist())

    reference_text = normalize_text(reference_text)
    hypothesis_text = normalize_text(hypothesis_text)

    wer_score = wer(reference_text, hypothesis_text)

    # Approximate hallucinations as words in hypothesis not in reference
    ref_words = set(reference_text.split())
    hyp_words = set(hypothesis_text.split())
    inserted_words = hyp_words - ref_words

    return wer_score, inserted_words, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--hyp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref_path = Path(args.ref)
    hyp_path = Path(args.hyp)
    out_path = Path(args.out)

    wer_score, inserted_words, df = compute_wer_from_files(ref_path, hyp_path)

    # Write metrics
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"WER: {wer_score:.4f}\n")
        if inserted_words:
            f.write("Potential Hallucinations (words in transcript not in reference):\n")
            for word in sorted(inserted_words):
                f.write(f"- {word}\n")
        else:
            f.write("No potential hallucinations detected.\n")

    # Mark hallucinations in transcript CSV
    if inserted_words:
        df['has_hallucination'] = df['Transcript'].apply(
            lambda text: any(word in inserted_words for word in str(text).lower().split()) if pd.notna(text) else False
        )
        flagged_csv_path = hyp_path.parent / f"{hyp_path.stem}_flagged{hyp_path.suffix}"
        df.to_csv(flagged_csv_path, index=False)
        print(f"Flagged transcript saved to {flagged_csv_path}")

    print(f"WER: {wer_score:.4f}")
    print(f"Detailed metrics saved to {out_path}")


if __name__ == "__main__":
    main()
    main()