# -*- coding: utf-8 -*-
"""
SamplePreprocessor_Hindi.py  —  MODERATE AUGMENTATION VERSION
----------------------------------------------------------
Balanced augmentation that prevents overfitting WITHOUT making images unreadable.

KEY PRINCIPLE: Model must be able to LEARN from augmented images!
- Apply only 1-2 augmentations per image (not 5+)
- Lower probabilities (20-30% instead of 40-60%)
- Gentler strength (±3° instead of ±5°)
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
    Preprocess with MODERATE augmentation.
    
    Strategy: Apply 1-2 augmentations per image, not 5+
    This prevents overfitting while keeping images learnable.
    """

    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]], dtype=np.uint8)

    # ── MODERATE DATA AUGMENTATION ───────────────────────────────────────────
    if dataAugmentation:
        
        # Choose 1-2 augmentations randomly (not all of them!)
        num_augs = random.choice([0, 1, 1, 2])  # Mostly 1, sometimes 2, rarely 0
        available_augs = ['rotate', 'brightness', 'blur', 'noise', 'scale', 'contrast']
        selected_augs = random.sample(available_augs, min(num_augs, len(available_augs)))
        
        # 1. ROTATION - More gentle (±3° instead of ±5°)
        if 'rotate' in selected_augs:
            angle = random.uniform(-3.0, 3.0)  # Reduced from ±5°
            h, w  = img.shape
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. BRIGHTNESS - Moderate variation
        if 'brightness' in selected_augs:
            delta = random.uniform(-20, 20)  # Reduced from ±30
            img   = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        
        # 3. CONTRAST - Gentle variation
        if 'contrast' in selected_augs:
            factor = random.uniform(0.8, 1.2)  # Reduced from 0.7-1.3
            img = np.clip((img.astype(np.float32) - 127.5) * factor + 127.5, 
                          0, 255).astype(np.uint8)

        # 4. BLUR - Mild blur only
        if 'blur' in selected_augs:
            kernel_size = 3  # Only size 3
            sigma = random.uniform(0.3, 1.0)  # Reduced from 0.5-2.0
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        # 5. NOISE - Light noise
        if 'noise' in selected_augs:
            std = random.uniform(3, 8)  # Reduced from 5-15
            noise = np.random.normal(0, std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. SCALING - Small scale variation
        if 'scale' in selected_augs:
            scale = random.uniform(0.9, 1.1)  # Reduced from 0.85-1.15
            h, w = img.shape
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            img = cv2.resize(img, (new_w, new_h))

    # ── Resize to target canvas ──────────────────────────────────────────────
    (wt, ht) = imgSize
    (h,  w)  = img.shape

    f       = max(w / wt, h / ht)
    new_w   = max(1, min(wt, int(w / f)))
    new_h   = max(1, min(ht, int(h / f)))

    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # White canvas
    canvas = np.ones([ht, wt], dtype=np.uint8) * 255
    canvas[0:new_h, 0:new_w] = img

    # ── Transpose for TF ──────────────────────────────────────────────────────
    transposed = cv2.transpose(canvas)

    # ── Normalization ─────────────────────────────────────────────────────────
    (m, s) = cv2.meanStdDev(transposed)
    m = m[0][0]
    s = s[0][0]
    normalised = (transposed - m) / (s + 1e-8)

    return normalised.astype(np.float32)
