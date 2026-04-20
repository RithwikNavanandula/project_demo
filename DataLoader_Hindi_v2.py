# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import gc
import numpy as np
import cv2
from collections import namedtuple
from datasets import load_dataset

from SamplePreprocessor_Hindi_v2 import preprocess

Batch = namedtuple("Batch", ["imgs", "gtTexts"])


class DataLoader:

    def __init__(self,
                 dataset_name: str,
                 images_dir: str,
                 batchSize: int,
                 imgSize: tuple,
                 maxTextLen: int,
                 charList: str,
                 dataAugmentation: bool = False,
                 val_split: float = 0.1):

        self.batchSize        = batchSize
        self.imgSize          = imgSize
        self.maxTextLen       = maxTextLen
        self.charList         = charList
        self.dataAugmentation = dataAugmentation
        self.images_dir       = images_dir
        self.charSet          = set(charList)

        print(f"[DataLoader] Loading dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split='train')
        dataset = dataset.shuffle(seed=42)

        split_idx = int(len(dataset) * (1 - val_split))

        self.train_dataset = dataset.select(range(split_idx))
        self.val_dataset   = dataset.select(range(split_idx, len(dataset)))

        print(f"[DataLoader] Train: {len(self.train_dataset)}")
        print(f"[DataLoader] Val:   {len(self.val_dataset)}")

        self._current_dataset = None
        self._idx = 0

    def _load_and_preprocess_image(self, img_path, augment):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros([self.imgSize[1], self.imgSize[0]], dtype=np.uint8)
        return preprocess(img, self.imgSize, dataAugmentation=augment)

    def trainSet(self, subset_size=None):
        dataset = self.train_dataset.shuffle(seed=None)

        if subset_size is not None:
            subset_size = min(subset_size, len(dataset))
            indices = np.random.choice(len(dataset), subset_size, replace=False)
            dataset = dataset.select(indices)
            print(f"[DataLoader] Training on {subset_size} samples")
        else:
            print(f"[DataLoader] Training on FULL dataset")

        self._current_dataset = dataset
        self._idx = 0

    def validationSet(self):
        self._current_dataset = self.val_dataset
        self._idx = 0

    def hasNext(self):
        return self._idx < len(self._current_dataset)

    def getIteratorInfo(self):
      """Return current batch index and total batches."""
      total = max(1, (len(self._current_dataset) + self.batchSize - 1) // self.batchSize)
      current = self._idx // self.batchSize + 1
      return (current, total)
      
    def getNext(self):
        end = min(self._idx + self.batchSize, len(self._current_dataset))
        batch_indices = range(self._idx, end)
        self._idx = end

        imgs = []
        gtTexts = []

        for idx in batch_indices:
          sample = self._current_dataset[idx]

          augment = (self._current_dataset != self.val_dataset)

          img = np.array(sample["image"].convert("L"))
          img = preprocess(img, self.imgSize, dataAugmentation=augment)

          imgs.append(img)
          gtTexts.append(sample['text'])
        return Batch(imgs=imgs, gtTexts=gtTexts)

    def cleanup(self):
        gc.collect()
