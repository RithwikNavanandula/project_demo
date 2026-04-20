# -*- coding: utf-8 -*-
"""
SamplePreprocessor_Hindi.py  —  Hindi OCR (Printed Text)
----------------------------------------------------------
Preprocesses printed Hindi text images.

SIMPLER than handwriting preprocessing:
  • No adaptive thresholding needed (synthetic text has clean contrast)
  • Minimal augmentation (printed text is consistent)
  • Simple normalization (no column-wise needed for printed text)
  • No elastic stretch

Target imgSize: (900, 64) to match dataset
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
        (width, height) of the target canvas — e.g. (900, 64).
    dataAugmentation : bool
        Apply random augmentations (minimal for printed text).

    Returns
    -------
    np.ndarray
        Float array of shape (width, height) — transposed for TF's
        time-major convention where axis-0 is the sequence (width) axis.
    """

    # ── Handle missing / corrupt images ──────────────────────────────────────
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]], dtype=np.uint8)

    # ── Data augmentation (MINIMAL for printed text) ─────────────────────────
    if dataAugmentation:
        
        # 1. Very slight rotation (±1°) — printed text is straight
        if random.random() < 0.3:
            angle = random.uniform(-1.0, 1.0)
            h, w  = img.shape
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)

        # 2. Slight brightness variation (printed text has good contrast)
        if random.random() < 0.3:
            delta = random.uniform(-15, 15)
            img   = np.clip(img.astype(np.float32) + delta, 0, 255
                            ).astype(np.uint8)

        # 3. Very mild blur (simulate lower resolution scans)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # NO elastic stretch - spacing is critical and consistent

    # ── Resize image proportionally into the target canvas ───────────────────
    (wt, ht) = imgSize                          # target width, height
    (h,  w)  = img.shape                        # source height, width

    # For this dataset, images are already 900×64, so this should be identity
    # But we keep the logic for robustness
    f       = max(w / wt, h / ht)
    new_w   = max(1, min(wt, int(w / f)))
    new_h   = max(1, min(ht, int(h / f)))

    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # White canvas (255 = background for printed text on white)
    canvas = np.ones([ht, wt], dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img

    # ── Transpose: (H × W) → (W × H) so that width becomes the time axis ─────
    transposed = cv2.transpose(canvas)          # shape: (wt, ht)

    # ── Simple global normalization ──────────────────────────────────────────
    # For printed text, global normalization is fine (consistent contrast)
    # No need for column-wise normalization like handwriting
    (m, s) = cv2.meanStdDev(transposed)
    m = m[0][0]
    s = s[0][0]
    normalised = (transposed - m) / s if s > 0 else (transposed - m)

    return normalised.astype(np.float32)
