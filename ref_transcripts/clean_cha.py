import re
import sys
from pathlib import Path


def clean_cha_text(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        # Keep only spoken utterance lines, e.g. *CHI:, *MOT:, *FAT:
        if not line.startswith("*"):
            continue

        # Remove speaker label, e.g. "*CHI:"
        line = re.sub(r"^\*[A-Z0-9]+:\s*", "", line)

        # Remove bullet/timestamp markers like 123_456 if present
        line = re.sub(r"\x15.*?\x15", "", line)

        # Remove bracketed CHAT annotations, e.g. [//], [=! laughs], [x 3]
        line = re.sub(r"\[[^\]]*\]", "", line)

        # Remove angle-bracket markup
        line = re.sub(r"<[^>]*>", "", line)

        # Remove common CHAT special tokens / events
        line = re.sub(r"&=\w+", "", line)          # &=laughs
        line = re.sub(r"&-\w+", "", line)          # &-um
        line = re.sub(r"&\+\w+", "", line)         # &+fragments
        line = re.sub(r"\bxxx\b|\byyy\b|\bwww\b", "", line, flags=re.IGNORECASE)

        # Remove filler punctuation / special symbols often used in CHAT
        line = re.sub(r"[<>⌈⌉⌊⌋„‡]", "", line)

        # Keep apostrophes, remove most other punctuation
        line = re.sub(r"[^\w\s']", " ", line)

        # Lowercase for WER normalization
        line = line.lower()

        # Collapse whitespace
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def convert_cha_to_txt(input_path: str, output_path: str) -> None:
    raw_text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_cha_text(raw_text)
    Path(output_path).write_text(cleaned_text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python cha_cleaner/clean_cha.py <path_to_file.cha>")

    input_cha = sys.argv[1]
    input_path = Path(input_cha)

    output_txt = input_path.parent / f"clean_{input_path.stem}.txt"
    convert_cha_to_txt(str(input_path), str(output_txt))
    print(f"Saved cleaned transcript to {output_txt}")