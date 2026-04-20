# -*- coding: utf-8 -*-
"""
DataLoader_Hindi.py  —  Hindi OCR (Printed Text) - COLAB OPTIMIZED
--------------------------------------------------------------------
Lazy-loading data loader for free-tier Colab (prevents memory crashes).

KEY OPTIMIZATIONS:
  ✅ Lazy loading - images loaded on-demand, not preloaded
  ✅ Reduced memory footprint - stores only paths, not images
  ✅ Batch-level preprocessing - process only what you need
  ✅ Memory cleanup - explicit garbage collection
"""

from __future__ import division
from __future__ import print_function

import os
import gc
import pandas as pd
import numpy as np
import cv2
from collections import namedtuple
from sklearn.model_selection import train_test_split

from SamplePreprocessor_Hindi import preprocess

Batch = namedtuple("Batch", ["imgs", "gtTexts"])


class DataLoader:
    """
    Memory-optimized DataLoader for Hindi printed text.
    
    Parameters
    ----------
    csv_path : str
        Path to data.csv file
    images_dir : str
        Path to folder containing images
    batchSize : int
        Number of samples per batch
    imgSize : tuple (width, height)
        Target image size after preprocessing
    maxTextLen : int
        Maximum label length; samples with longer text are skipped
    charList : str
        String of all valid characters
    dataAugmentation : bool
        Whether to apply augmentation
    val_split : float
        Fraction of data to use for validation (default 0.1)
    max_samples : int or None
        Limit dataset size for testing (use None for full dataset)
    """

    def __init__(self,
                 csv_path: str,
                 images_dir: str,
                 batchSize: int,
                 imgSize: tuple,
                 maxTextLen: int,
                 charList: str,
                 dataAugmentation: bool = False,
                 val_split: float = 0.1,
                 max_samples: int = None):

        self.batchSize        = batchSize
        self.imgSize          = imgSize
        self.maxTextLen       = maxTextLen
        self.charList         = charList
        self.dataAugmentation = dataAugmentation
        self.images_dir       = images_dir

        self.charSet = set(charList)

        print(f"[DataLoader] Loading dataset from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Limit samples for testing on free tier
        if max_samples is not None:
            df = df.head(max_samples)
            print(f"[DataLoader] ⚠️  Limited to {max_samples} samples (testing mode)")
        
        print(f"[DataLoader] Total samples in CSV: {len(df)}")
        
        # Split into train/val
        train_df, val_df = train_test_split(
            df, test_size=val_split, random_state=42
        )
        
        print(f"[DataLoader] Train samples: {len(train_df)}")
        print(f"[DataLoader] Val samples:   {len(val_df)}")

        # Load samples (ONLY PATHS, NOT IMAGES)
        self.trainSamples      = self._load_split(train_df)
        self.validationSamples = self._load_split(val_df)

        # Collect words for corpus (without loading images)
        self.trainWords      = self._collect_words(self.trainSamples)
        self.validationWords = self._collect_words(self.validationSamples)

        print(f"[DataLoader] Ready: Train={len(self.trainSamples)}  "
              f"Val={len(self.validationSamples)}")
        print(f"[DataLoader] 💾 Memory: Lazy loading enabled (images loaded on-demand)")

        self._currentSamples: list = []
        self._idx: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_valid(self, text: str) -> bool:
        """Return True if every character in text is in our charList."""
        if len(text) == 0 or len(text) > self.maxTextLen:
            return False
        return all(c in self.charSet for c in text)

    def _load_split(self, df) -> list:
        """
        Build list of (image_path, text) tuples.
        
        ⚠️ CRITICAL CHANGE: We store PATHS, not preprocessed images!
        This prevents memory crashes on free-tier Colab.
        """
        samples = []
        skipped = 0

        for idx, row in df.iterrows():
            text = row['text']
            
            if not self._is_valid(text):
                skipped += 1
                continue

            # Build image path
            img_path = os.path.join(self.images_dir, row['image_file'])
            
            # Check if file exists
            if not os.path.exists(img_path):
                print(f"[DataLoader] Warning: Image not found: {img_path}")
                skipped += 1
                continue
            
            # Store PATH, not image (lazy loading)
            samples.append((img_path, text))

        if skipped:
            print(f"[DataLoader] Skipped {skipped} samples "
                  f"(OOV chars, missing files, or length > {self.maxTextLen})")
        
        return samples

    @staticmethod
    def _collect_words(samples: list) -> list:
        """Extract words from text labels (no image loading needed)."""
        words = []
        for _, text in samples:
            words.extend(text.split())
        return words

    def _load_and_preprocess_image(self, img_path: str, augment: bool) -> np.ndarray:
        """
        Load and preprocess a single image on-demand.
        
        This is called only when a batch is requested, not during initialization.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"[DataLoader] Warning: Cannot read image: {img_path}")
            # Return empty image as fallback
            img = np.zeros([self.imgSize[1], self.imgSize[0]], dtype=np.uint8)
        
        # Preprocess
        processed = preprocess(img, self.imgSize,
                               dataAugmentation=(augment and self.dataAugmentation))
        return processed

    # ── Public split selectors ────────────────────────────────────────────────

    def trainSet(self):
        """Switch iterator to training split (shuffled)."""
        self._currentSamples = self.trainSamples.copy()
        np.random.shuffle(self._currentSamples)
        self._idx = 0

    def validationSet(self):
        """Switch iterator to validation split."""
        self._currentSamples = self.validationSamples
        self._idx = 0

    # ── Iterator API ──────────────────────────────────────────────────────────

    def hasNext(self) -> bool:
        return self._idx < len(self._currentSamples)

    def getIteratorInfo(self) -> tuple:
        """Returns (currentBatch, totalBatches) — 1-indexed."""
        total   = max(1, (len(self._currentSamples) + self.batchSize - 1)
                      // self.batchSize)
        current = self._idx // self.batchSize + 1
        return (current, total)

    def getNext(self) -> Batch:
        """
        Return next Batch of preprocessed images and ground-truth texts.
        
        ⚠️ CRITICAL: Images are loaded and preprocessed HERE, not during init.
        This prevents memory crashes by processing only one batch at a time.
        """
        end   = min(self._idx + self.batchSize, len(self._currentSamples))
        chunk = self._currentSamples[self._idx:end]
        self._idx = end

        # Extract paths and texts
        img_paths = [s[0] for s in chunk]
        gtTexts   = [s[1] for s in chunk]
        
        # Load and preprocess images ON-DEMAND
        imgs = []
        for img_path in img_paths:
            # Determine if this is training data (for augmentation)
            is_training = (self._currentSamples == self.trainSamples)
            processed_img = self._load_and_preprocess_image(img_path, augment=is_training)
            imgs.append(processed_img)
        
        return Batch(imgs=imgs, gtTexts=gtTexts)

    def cleanup(self):
        """Explicit memory cleanup (call between train/val splits)."""
        gc.collect()
