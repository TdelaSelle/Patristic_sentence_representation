"""
Corpus preparation for unsupervised sentence embedding training.

Pipeline:
  TXT files → ClauseSegmenter → lowercase + strip non-alphanumeric → one line per segment
  CSV files → 'text' column   → lowercase + strip non-alphanumeric → one line per row
"""
    
import re
import csv
import sys
from pathlib import Path

# Make segmenter importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))
from processing_files import segmenter
# from segmenter import ClauseSegmenter

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = Path("preprocessed_corpus/Aug_trin")
OUTPUT_FILE = Path("preprocessed_corpus/Aug_trin.txt")

Latin_linguistic_markers = {
        "et", "sed", "quia", "quoniam", "autem",
        "enim", "vero", "nam", "sicut", "igitur", "ergo",
        "atque", "ac", "nec", "neque", "vel", "aut",
        "ut", "cum", "si", "nisi", "tamen",
    }
Greek_linguistic_markers = {
        "καί", "δέ", "ἀλλά", "γάρ", "μέν", "τε",
        "ὅτι", "διότι", "ἐπεί", "ἐπειδή",
        "οὖν", "ἄρα", "διό", "τοίνυν",
        "ὡς", "ὥσπερ", "καθώς",
        "ἵνα", "ὅπως", "ὥστε",
        "ὅτε", "ὅταν",
        "εἰ", "ἐάν",
        "ἤ",
        "οὐδέ", "μηδέ", "οὔτε", "μήτε",
        "ὅμως", "μέντοι", "πλήν",
    }


# ClauseSegmenter settings
SEGMENTER = segmenter.ClauseSegmenter(
    min_words=7,
    max_words=100,
    merge_orphans=True,
    linguistic_markers=Latin_linguistic_markers,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Lowercase and remove every character that is not alphanumeric or whitespace."""
    text = text.lower()
    # text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
    text = re.sub(r"\s+", " ", text)        # collapse runs of whitespace
    return text.strip()


def process_txt(path: Path):
    """Read a plain-text file, segment it, clean each segment, yield non-empty lines."""
    raw = path.read_text(encoding="utf-8")
    for segment in SEGMENTER.segment(raw):
        line = clean(segment)
        if line:
            yield line


def process_csv(path: Path):
    """Read the 'text' column of a CSV file, clean each value, yield non-empty lines."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "text" not in (reader.fieldnames or []):
            print(f"  [skip] no 'text' column in {path.name}")
            return
        for row in reader:
            line = clean(row["text"])
            if line:
                yield line


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    print(f"Found {len(txt_files)} TXT file(s) and {len(csv_files)} CSV file(s) in {INPUT_DIR}")

    total_lines = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in txt_files:
            print(f"  Processing TXT: {path.name}")
            count = 0
            for line in process_txt(path):
                out.write(line + "\n")
                count += 1
            total_lines += count
            print(f"    → {count} segments")

        for path in csv_files:
            print(f"  Processing CSV: {path.name}")
            count = 0
            for line in process_csv(path):
                out.write(line + "\n")
                count += 1
            total_lines += count
            print(f"    → {count} rows")

    print(f"\nDone. {total_lines} total lines written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
