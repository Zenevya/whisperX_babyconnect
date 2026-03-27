import re
from pathlib import Path
import pandas as pd
from jiwer import wer


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # keep apostrophes, remove most punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# reference transcript from cleaned .cha
reference_path = Path("ref_transcripts/clean_CD15_020517a.txt")
reference_text = reference_path.read_text(encoding="utf-8", errors="ignore")

# whisperx transcript csv
csv_path = Path("output_transcripts/transcriptions_CD15_020517a_optimized.csv")
df = pd.read_csv(csv_path)

# combine all transcript segments into one string
hypothesis_text = " ".join(df["Transcript"].dropna().astype(str).tolist())

# normalize both
reference_text = normalize_text(reference_text)
hypothesis_text = normalize_text(hypothesis_text)

# compute WER
score = wer(reference_text, hypothesis_text)

print("REFERENCE:")
print(reference_text[:500], "\n")
print("HYPOTHESIS:")
print(hypothesis_text[:500], "\n")
print(f"WER: {score:.4f}")