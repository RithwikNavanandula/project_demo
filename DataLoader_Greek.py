# -*- coding: utf-8 -*-
"""
DataLoader.py  —  Greek Sentence-Level HTR
-------------------------------------------
Loads the rithwikn/greek_combined_dataset from HuggingFace.
Each sample: PIL image  +  cleaned transcription string.

Batch namedtuple mirrors the original Batch(imgs, gtTexts)
so Model.py and main.py need zero changes to their batch API.
"""

from __future__ import division
from __future__ import print_function

import re
import numpy as np
import cv2
from collections import namedtuple
from datasets import load_dataset
from PIL import Image

from SamplePreprocessor import preprocess

# ── same Batch interface as original code ─────────────────────────────────────
Batch = namedtuple("Batch", ["imgs", "gtTexts"])

# ── annotation markers to strip from transcriptions ───────────────────────────
REMOVE_PATTERNS = [
    r"<\+\+>",
    r"<\+>",
    r"\{\d+\}",
    r"\[\d+\]",
]

def clean_transcription(text: str) -> str:
    """Strip manuscript annotation markers; collapse whitespace."""
    for pat in REMOVE_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


class DataLoader:
    """
    Loads greek_combined_dataset from HuggingFace and serves batches.

    Parameters
    ----------
    hf_dataset_name : str
        HuggingFace dataset repo id.
    batchSize : int
        Number of samples per batch.
    imgSize : tuple (width, height)
        Target image size after preprocessing.
    maxTextLen : int
        Maximum label length; samples with longer transcriptions are skipped.
    charList : str
        String of all valid characters (built by build_charlist.py).
    dataAugmentation : bool
        Whether to apply augmentation to training images.
    """

    def __init__(self,
                 hf_dataset_name: str,
                 batchSize: int,
                 imgSize: tuple,
                 maxTextLen: int,
                 charList: str,
                 dataAugmentation: bool = False):

        self.batchSize        = batchSize
        self.imgSize          = imgSize
        self.maxTextLen       = maxTextLen
        self.charList         = charList
        self.dataAugmentation = dataAugmentation

        self.charSet = set(charList)

        print(f"[DataLoader] Loading dataset: {hf_dataset_name}")
        raw = load_dataset(hf_dataset_name)

        self.trainSamples      = self._load_split(raw, "train",      augment=True)
        self.validationSamples = self._load_split(raw, "validation", augment=False)
        self.testSamples       = self._load_split(raw, "test",       augment=False)

        # collect all words for corpus (used if WordBeamSearch is enabled)
        self.trainWords      = self._collect_words(self.trainSamples)
        self.validationWords = self._collect_words(self.validationSamples)

        print(f"[DataLoader] Train={len(self.trainSamples)}  "
              f"Val={len(self.validationSamples)}  "
              f"Test={len(self.testSamples)}")

        self._currentSamples: list = []
        self._idx: int = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _pil_to_cv2_gray(self, pil_img) -> np.ndarray:
        """Convert a PIL image (any mode) to an OpenCV grayscale array."""
        pil_img = pil_img.convert("L")          # grayscale
        return np.array(pil_img, dtype=np.uint8)

    def _is_valid(self, text: str) -> bool:
        """Return True if every character in text is in our charList."""
        if len(text) == 0 or len(text) > self.maxTextLen:
            return False
        return all(c in self.charSet for c in text)

    def _load_split(self, raw_dataset, split_name: str,
                    augment: bool) -> list:
        """
        Build a list of (preprocessed_img_array, clean_transcription) tuples
        for one split.  Samples with out-of-vocabulary chars or overlong
        transcriptions are skipped with a warning count.
        """
        if split_name not in raw_dataset:
            return []

        samples   = []
        skipped   = 0

        for row in raw_dataset[split_name]:
            text = clean_transcription(row["transcription"])

            if not self._is_valid(text):
                skipped += 1
                continue

            # Convert HuggingFace image (PIL) → grayscale numpy array
            pil_img = row["image"]
            cv2_img = self._pil_to_cv2_gray(pil_img)

            # Preprocess: resize to target imgSize, normalise
            processed = preprocess(cv2_img, self.imgSize,
                                   dataAugmentation=(augment and self.dataAugmentation))
            samples.append((processed, text))

        if skipped:
            print(f"[DataLoader] '{split_name}': skipped {skipped} samples "
                  f"(OOV chars or length > {self.maxTextLen})")
        return samples

    @staticmethod
    def _collect_words(samples: list) -> list:
        words = []
        for _, text in samples:
            words.extend(text.split())
        return words

    # ── public split selectors (match original DataLoader API) ────────────────

    def trainSet(self):
        """Switch iterator to training split (shuffled)."""
        self._currentSamples = self.trainSamples.copy()
        np.random.shuffle(self._currentSamples)
        self._idx = 0

    def validationSet(self):
        """Switch iterator to validation split."""
        self._currentSamples = self.validationSamples
        self._idx = 0

    def testSet(self):
        """Switch iterator to test split."""
        self._currentSamples = self.testSamples
        self._idx = 0

    # ── iterator API (match original DataLoader API) ──────────────────────────

    def hasNext(self) -> bool:
        return self._idx < len(self._currentSamples)

    def getIteratorInfo(self) -> tuple:
        """Returns (currentBatch, totalBatches) — 1-indexed."""
        total = max(1, (len(self._currentSamples) + self.batchSize - 1)
                    // self.batchSize)
        current = self._idx // self.batchSize + 1
        return (current, total)

    def getNext(self) -> Batch:
        """Return next Batch of preprocessed images and ground-truth texts."""
        end  = min(self._idx + self.batchSize, len(self._currentSamples))
        chunk = self._currentSamples[self._idx:end]
        self._idx = end

        imgs    = [s[0] for s in chunk]
        gtTexts = [s[1] for s in chunk]
        return Batch(imgs=imgs, gtTexts=gtTexts)
