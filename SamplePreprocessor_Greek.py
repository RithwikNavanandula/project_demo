# -*- coding: utf-8 -*-
"""
SamplePreprocessor.py  —  Greek Sentence-Level HTR
----------------------------------------------------
Preprocesses a grayscale image into a fixed-size, normalised array
ready for the CRNN model.

Key changes from the word-level version:
  • Target imgSize is now (1024, 64) instead of (128, 32)
  • Data augmentation is fully implemented (was a stub before)
  • White-fill canvas (255) for handwriting images
"""

from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2


def preprocess(img: np.ndarray,
               imgSize: tuple,
               dataAugmentation: bool = False) -> np.ndarray:
    """
    Resize *img* into a canvas of shape (imgSize[1], imgSize[0]),
    optionally augment, then transpose and normalise for TensorFlow.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (H × W, uint8).
    imgSize : tuple
        (width, height) of the target canvas — e.g. (1024, 64).
    dataAugmentation : bool
        Apply random augmentations (use True only during training).

    Returns
    -------
    np.ndarray
        Float array of shape (width, height) — transposed for TF's
        time-major convention where axis-0 is the sequence (width) axis.
    """

    # ── handle missing / corrupt images ──────────────────────────────────────
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]], dtype=np.uint8)

    # ── optional data augmentation (training only) ────────────────────────────
    if dataAugmentation:

        # 1. Slight random rotation (±2°) — simulates tilted handwriting
        if random.random() < 0.5:
            angle = random.uniform(-2.0, 2.0)
            h, w  = img.shape
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)

        # 2. Random brightness shift (±20 grey levels) — pen pressure variation
        if random.random() < 0.5:
            delta = random.uniform(-20, 20)
            img   = np.clip(img.astype(np.float32) + delta, 0, 255
                            ).astype(np.uint8)

        # 3. Mild Gaussian blur — simulates ink bleed / low-res scans
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # 4. Random horizontal elastic stretch (±10 %) — writing width variation
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            h, w  = img.shape
            new_w = max(1, int(w * scale))
            img   = cv2.resize(img, (new_w, h))

    # ── resize image proportionally into the target canvas ───────────────────
    (wt, ht) = imgSize                          # target width, height
    (h,  w)  = img.shape                        # source height, width

    # Scale factor that makes the image fit inside (wt × ht)
    f       = max(w / wt, h / ht)
    new_w   = max(1, min(wt, int(w / f)))
    new_h   = max(1, min(ht, int(h / f)))

    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # White canvas (255 = background for handwriting on white paper)
    canvas = np.ones([ht, wt], dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img

    # ── transpose: (H × W) → (W × H) so that width becomes the time axis ─────
    # TensorFlow's RNN processes axis-0 as time; after transpose axis-0 = width
    transposed = cv2.transpose(canvas)          # shape: (wt, ht)

    # ── per-image zero-mean / unit-std normalisation ──────────────────────────
    (m, s) = cv2.meanStdDev(transposed)
    m = m[0][0]
    s = s[0][0]
    normalised = (transposed - m) / s if s > 0 else (transposed - m)

    return normalised.astype(np.float32)
