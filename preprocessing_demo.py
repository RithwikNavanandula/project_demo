"""
preprocessing_demo.py
=====================
Demonstrates all preprocessing steps applied in the CRNN+CTC pipeline
on a single input image.  Run this script to produce a step-by-step
visualisation figure (saved as preprocessing_steps.png) suitable for
inclusion in the project report.

Usage:
    python preprocessing_demo.py --image /absolute/or/relative/path/to/local_image.png

Dependencies:
    pip install opencv-python numpy matplotlib Pillow
"""

import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFilter
import os

# ── Configuration (must match training settings) ─────────────────────────────
IMG_H_LINE  = 64    # target height for line models
IMG_W_LINE  = 1024  # maximum width for line models
IMG_H_WORD  = 32    # target height for word model
IMG_W_WORD  = 128   # maximum width for word model


# ── Step functions ────────────────────────────────────────────────────────────

def step0_load(path: str) -> np.ndarray:
    """Load image in BGR (OpenCV default)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def step1_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR → single-channel grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def step2_resize(gray: np.ndarray,
                 target_h: int = IMG_H_LINE,
                 target_w: int = IMG_W_LINE) -> np.ndarray:
    """
    Aspect-ratio-preserving resize.
    1. Scale height to target_h, keep aspect ratio.
    2. Pad/crop width to target_w with white (255) pixels.
    """
    h, w = gray.shape
    # scale so height == target_h
    scale  = target_h / h
    new_w  = int(w * scale)
    resized = cv2.resize(gray, (new_w, target_h),
                         interpolation=cv2.INTER_AREA)
    # build white canvas
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    paste_w = min(new_w, target_w)
    canvas[:, :paste_w] = resized[:, :paste_w]
    return canvas


def step3_normalise(canvas: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1] float32."""
    return (canvas.astype(np.float32) / 255.0)


def step4_augment_rotation(norm: np.ndarray,
                            angle_deg: float = 1.5) -> np.ndarray:
    """
    Random rotation ±2° (shown at fixed +1.5° for demo).
    Input is float [0,1]; white background = 1.0.
    """
    uint8 = (norm * 255).astype(np.uint8)
    h, w  = uint8.shape
    cx, cy = w // 2, h // 2
    M  = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(uint8, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
    return rotated.astype(np.float32) / 255.0


def step5_augment_brightness(norm: np.ndarray,
                              delta: int = 18) -> np.ndarray:
    """
    Brightness shift ±20 (shown at +18 for demo).
    Clips to [0, 255] before renormalising.
    """
    uint8   = (norm * 255).astype(np.uint8)
    shifted = np.clip(uint8.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    return shifted.astype(np.float32) / 255.0


def step6_augment_blur(norm: np.ndarray) -> np.ndarray:
    """Gaussian blur with 3×3 kernel (simulates ink bleed)."""
    uint8   = (norm * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(uint8, (3, 3), sigmaX=0.8)
    return blurred.astype(np.float32) / 255.0


def step7_augment_stretch(norm: np.ndarray,
                           factor: float = 1.08) -> np.ndarray:
    """
    Horizontal stretch ±10% (shown at +8% for demo).
    Resizes width by factor then re-pads to original width.
    """
    uint8  = (norm * 255).astype(np.uint8)
    h, w   = uint8.shape
    new_w  = int(w * factor)
    stretched = cv2.resize(uint8, (new_w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    paste  = min(new_w, w)
    canvas[:, :paste] = stretched[:, :paste]
    return canvas.astype(np.float32) / 255.0


def step8_transpose(norm: np.ndarray) -> np.ndarray:
    """
    Transpose so width (time axis) becomes axis-0.
    Shape: (H, W) → (W, H) — this is what the RNN receives as a sequence.
    Shown transposed back for display purposes.
    """
    return norm.T  # (W, H)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(steps: list, out_path: str = "preprocessing_steps.png"):
    """
    Plot all preprocessing steps in a grid.
    steps: list of (title, image_array, cmap)
    """
    n     = len(steps)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 2.8 * nrows),
                             facecolor="white")
    axes = axes.flatten()

    for i, (title, img, cmap) in enumerate(steps):
        ax = axes[i]
        if img.ndim == 2:
            vmin, vmax = (0, 255) if img.dtype == np.uint8 else (0.0, 1.0)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    # hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # add step arrows annotation
    fig.suptitle(
        "CRNN+CTC Preprocessing Pipeline — Step-by-Step Visualisation\n"
        "(Augmentation steps shown for illustration; applied with probability < 1.0 during training)",
        fontsize=12, y=1.01
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    print(f"[INFO] Saved to: {out_path}")
    plt.close()


def _to_uint8_for_save(img: np.ndarray) -> np.ndarray:
    """Convert float/gray/BGR arrays into savable uint8 arrays."""
    if img.dtype == np.uint8:
        return img
    clipped = np.clip(img, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def save_step_images(steps: list, out_dir: str):
    """Save each preprocessing step as a separate image file."""
    os.makedirs(out_dir, exist_ok=True)

    for idx, (title, img, _) in enumerate(steps):
        slug = title.lower()
        slug = slug.replace("\n", " ")
        slug = "".join(ch if ch.isalnum() else "_" for ch in slug)
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug.strip("_")
        filename = f"step_{idx:02d}_{slug}.png"
        path = os.path.join(out_dir, filename)

        if img.ndim == 3:
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, _to_uint8_for_save(img))

    print(f"[INFO] Saved {len(steps)} separate step images in: {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise all CRNN+CTC preprocessing steps on a single image."
    )
    parser.add_argument("--image",  type=str,
                        required=True,
                        help="Path to a local input image (line or word crop).")
    parser.add_argument("--mode",   type=str,
                        choices=["line", "word"], default="line",
                        help="'line' uses 64×1024; 'word' uses 32×128.")
    parser.add_argument("--output", type=str,
                        default="preprocessing_steps.png",
                        help="Output figure path.")
    parser.add_argument("--output-dir", type=str,
                        default="preprocessing_outputs",
                        help="Directory to save each preprocessing step as separate images.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(
            f"Input image not found: {args.image}\n"
            f"Provide a valid local image path with --image"
        )

    # ── Select dimensions ────────────────────────────────────────────────────
    tgt_h = IMG_H_LINE if args.mode == "line" else IMG_H_WORD
    tgt_w = IMG_W_LINE if args.mode == "line" else IMG_W_WORD

    # ── Run pipeline ─────────────────────────────────────────────────────────
    print("[INFO] Running preprocessing pipeline …")

    s0_bgr   = step0_load(args.image)
    s1_gray  = step1_grayscale(s0_bgr)
    s2_resz  = step2_resize(s1_gray, tgt_h, tgt_w)
    s3_norm  = step3_normalise(s2_resz)
    s4_rot   = step4_augment_rotation(s3_norm)
    s5_bri   = step5_augment_brightness(s3_norm)
    s6_blur  = step6_augment_blur(s3_norm)
    s7_str   = step7_augment_stretch(s3_norm)
    s8_trans = step8_transpose(s3_norm)   # transpose of normalised (no augment)

    # Print shapes
    print(f"  Original   : {s0_bgr.shape}")
    print(f"  Grayscale  : {s1_gray.shape}")
    print(f"  After resize (H={tgt_h}, W={tgt_w}): {s2_resz.shape}")
    print(f"  Normalised : {s3_norm.shape}  dtype={s3_norm.dtype}  "
          f"min={s3_norm.min():.3f} max={s3_norm.max():.3f}")
    print(f"  Transposed (→ RNN input): {s8_trans.shape}")

    # ── Collect steps for display ────────────────────────────────────────────
    steps = [
        ("Step 0: Original Input",
         s0_bgr,                             "gray"),
        ("Step 1: Grayscale Conversion",
         s1_gray,                            "gray"),
        (f"Step 2: Aspect-Ratio Resize\n(H={tgt_h}, W={tgt_w}, white padding)",
         s2_resz,                            "gray"),
        ("Step 3: Pixel Normalisation\n[0, 255] → [0.0, 1.0]",
         s3_norm,                            "gray"),
        ("Step 4 (Aug): Random Rotation ±2°\n(prob=0.5 during training)",
         s4_rot,                             "gray"),
        ("Step 5 (Aug): Brightness Shift ±20\n(prob=0.5 during training)",
         s5_bri,                             "gray"),
        ("Step 6 (Aug): Gaussian Blur 3×3\n(prob=0.3 during training)",
         s6_blur,                            "gray"),
        ("Step 7 (Aug): Horizontal Stretch ±10%\n(prob=0.3 during training)",
         s7_str,                             "gray"),
        ("Step 8: Transpose (H,W)→(W,H)\nWidth becomes RNN time-axis",
         s8_trans,                           "gray"),
    ]

    visualise(steps, out_path=args.output)
    save_step_images(steps, out_dir=args.output_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
