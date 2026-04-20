# -*- coding: utf-8 -*-
"""
build_charlist.py
-----------------
Run this ONCE before training to auto-build charList.txt
from the actual greek_combined_dataset transcriptions.

Usage:
    python build_charlist.py

Output:
    ../model_sentence/charList.txt
"""

import os
import re
from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
DATASET_NAME  = "rithwikn/greek_combined_dataset"
OUTPUT_PATH   = "../model_sentence/charList.txt"

# Annotation markers to REMOVE from transcriptions before building charlist.
# These are manuscript annotation tags, not actual handwritten text.
REMOVE_PATTERNS = [
    r"<\+\+>",          # <++>
    r"<\+>",            # <+>
    r"\{\d+\}",         # {1}, {2}, ...
    r"\[\d+\]",         # [30], [31], ...
]

def clean_transcription(text: str) -> str:
    """Remove annotation markers from a transcription string."""
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text).strip()
    return text

def build_charlist():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    all_chars = set()

    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        print(f"  Processing split: {split_name} ({len(dataset[split_name])} rows)")
        for row in dataset[split_name]:
            text = clean_transcription(row["transcription"])
            all_chars.update(set(text))

    # Remove empty string if present
    all_chars.discard("")

    # Sort: put space first (after blank '-'), then sort rest unicode order
    sorted_chars = sorted(all_chars, key=lambda c: (c == " ", ord(c)))

    # Build final charlist:
    # Index 0 = '-' (CTC blank token) — MUST be first
    # Index 1 = ' ' (space — critical for sentences)
    # Index 2+ = all other characters in unicode order
    space_char = " "
    other_chars = [c for c in sorted_chars if c != space_char]

    final_charlist = "-" + space_char + "".join(other_chars)

    # Make output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_charlist)

    print(f"\n✅ charList.txt saved to: {OUTPUT_PATH}")
    print(f"   Total characters: {len(final_charlist)}")
    print(f"   Index 0 (CTC blank): '-'")
    print(f"   Index 1 (space):     ' '")
    print(f"   Sample chars: {final_charlist[:60]}...")
    print(f"\n   Full charlist:\n   {final_charlist}")

if __name__ == "__main__":
    build_charlist()
