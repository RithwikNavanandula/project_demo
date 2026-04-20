# -*- coding: utf-8 -*-
"""
build_charlist_hindi.py
-----------------------
Build charList.txt from Hindi OCR dataset.

Extracts all unique characters from the 'text' column in data.csv
and creates a character list for CTC training.

Usage:
    python build_charlist_hindi.py --csv path/to/data.csv

Output:
    ../model_hindi/charList.txt
"""

import os
import argparse
import pandas as pd


def build_charlist(csv_path: str, output_path: str):
    """
    Extract all unique characters from CSV and build charList.txt.
    
    Format:
        Index 0: '-' (CTC blank token)
        Index 1: ' ' (space - critical for word boundaries)
        Index 2+: All other characters in unicode order
    """
    print(f"Loading dataset from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Extract all unique characters
    all_chars = set()
    
    for text in df['text']:
        all_chars.update(set(text))
    
    # Remove empty string if present
    all_chars.discard("")
    
    print(f"\nFound {len(all_chars)} unique characters")
    
    # Separate space from other characters
    space_char = " "
    other_chars = [c for c in all_chars if c != space_char]
    
    # Sort other characters by unicode value
    other_chars.sort(key=lambda c: ord(c))
    
    # Build final charlist:
    # Index 0 = '-' (CTC blank token) — MUST be first
    # Index 1 = ' ' (space — critical for word boundaries)
    # Index 2+ = all other characters in unicode order
    final_charlist = "-" + space_char + "".join(other_chars)
    
    # Make output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_charlist)
    
    print(f"\n✅ charList.txt saved to: {output_path}")
    print(f"   Total characters: {len(final_charlist)}")
    print(f"   Index 0 (CTC blank): '-'")
    print(f"   Index 1 (space):     ' '")
    print(f"\n   First 50 characters:")
    print(f"   {repr(final_charlist[:50])}")
    print(f"\n   Sample Devanagari characters:")
    
    # Show some sample Devanagari characters
    devanagari_chars = [c for c in other_chars if '\u0900' <= c <= '\u097F']
    print(f"   {repr(''.join(devanagari_chars[:30]))}")
    
    # Statistics
    print(f"\n📊 Character set statistics:")
    print(f"   Devanagari consonants/vowels: {len(devanagari_chars)}")
    print(f"   Digits: {sum(c.isdigit() for c in other_chars)}")
    print(f"   Punctuation: {sum(not c.isalnum() for c in other_chars)}")
    
    # Show character distribution
    print(f"\n   Unicode ranges:")
    ranges = {
        'Devanagari': (0x0900, 0x097F),
        'ASCII': (0x0000, 0x007F),
        'Latin Extended': (0x0080, 0x00FF),
    }
    
    for name, (start, end) in ranges.items():
        count = sum(start <= ord(c) <= end for c in other_chars)
        if count > 0:
            print(f"   {name}: {count} characters")


def main():
    parser = argparse.ArgumentParser(
        description="Build character list for Hindi OCR dataset")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to data.csv file")
    parser.add_argument("--output", type=str, 
                        default="../model_hindi/charList.txt",
                        help="Output path for charList.txt")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        return
    
    build_charlist(args.csv, args.output)


if __name__ == "__main__":
    main()
