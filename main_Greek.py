# -*- coding: utf-8 -*-
"""
main.py  —  Greek Sentence-Level HTR
--------------------------------------
Entry point for training, validation, and inference.

Usage
-----
# Step 0: build charList (run once)
    python build_charlist.py

# Train from scratch
    python main.py --train

# Validate only
    python main.py --validate

# Infer on a single image (best-path decoder)
    python main.py --infer --image path/to/image.png

# Infer with beam search
    python main.py --infer --image path/to/image.png --beamsearch
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import codecs

import cv2
import numpy as np
import tensorflow as tf
import editdistance

from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

tf.compat.v1.disable_eager_execution()

# ── paths ─────────────────────────────────────────────────────────────────────
class FilePaths:
    fnCharList  = "model_greek/charList.txt"
    fnAccuracy  = "model_greek/accuracy.txt"
    fnCorpus    = "model_greek/corpus.txt"

HF_DATASET = "rithwikn/greek_combined_dataset"


# ── training ──────────────────────────────────────────────────────────────────
def train(model: Model, loader: DataLoader):
    """Train the model with early stopping on validation CER."""
    epoch              = model.lastEpoch
    bestCharErrorRate  = float("inf")
    noImprovementSince = 0
    earlyStopping      = 10          # stop after 10 epochs without improvement

    while True:
        epoch += 1
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        # ── train one epoch ───────────────────────────────────────────────────
        print("Training...")
        loader.trainSet()
        epoch_loss = []
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch    = loader.getNext()
            loss     = model.trainBatch(batch)
            epoch_loss.append(loss)
            if iterInfo[0] % 10 == 0 or iterInfo[0] == iterInfo[1]:
                print(f"  Batch {iterInfo[0]:>4}/{iterInfo[1]}  loss={loss:.4f}")

        print(f"  Mean epoch loss: {np.mean(epoch_loss):.4f}")

        # ── validate ─────────────────────────────────────────────────────────
        charErrorRate = validate(model, loader)

        # ── checkpoint logic ─────────────────────────────────────────────────
        if charErrorRate < bestCharErrorRate:
            print(f"✅ CER improved {bestCharErrorRate*100:.2f}% → "
                  f"{charErrorRate*100:.2f}%  — saving model")
            bestCharErrorRate  = charErrorRate
            noImprovementSince = 0
            model.save(epoch)
            with open(FilePaths.fnAccuracy, "w", encoding="utf-8") as f:
                f.write(f"Best validation CER: {charErrorRate*100:.2f}%  "
                        f"(epoch {epoch})\n")
        else:
            print(f"  CER did not improve ({charErrorRate*100:.2f}% vs "
                  f"best {bestCharErrorRate*100:.2f}%)")
            noImprovementSince += 1

        if noImprovementSince >= earlyStopping:
            print(f"\nNo improvement for {earlyStopping} epochs — "
                  f"training stopped.")
            break


# ── validation ────────────────────────────────────────────────────────────────
def validate(model: Model, loader: DataLoader) -> float:
    """Run validation set; return Character Error Rate."""
    print("Validating...")
    loader.validationSet()

    numCharErr   = 0
    numCharTotal = 0
    numWordOK    = 0
    numWordTotal = 0

    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        batch    = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        for i in range(len(recognized)):
            gt   = batch.gtTexts[i]
            pred = recognized[i]

            numWordOK    += 1 if gt == pred else 0
            numWordTotal += 1

            dist          = editdistance.eval(pred, gt)
            numCharErr   += dist
            numCharTotal += len(gt)

            status = "[OK]" if dist == 0 else f"[ERR:{dist}]"
            print(f"  {status}  GT: '{gt}'")
            print(f"         PRD: '{pred}'")

    charErrorRate = numCharErr  / max(numCharTotal, 1)
    wordAccuracy  = numWordOK   / max(numWordTotal, 1)
    print(f"\nValidation CER:      {charErrorRate*100:.2f}%")
    print(f"Validation Word Acc: {wordAccuracy*100:.2f}%")
    return charErrorRate


# ── single-image inference ────────────────────────────────────────────────────
def infer(model: Model, img_path: str) -> str:
    """Recognise text in a single image file."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    processed = preprocess(img, Model.imgSize, dataAugmentation=False)
    batch     = Batch(imgs=[processed], gtTexts=None)

    (recognized, probability) = model.inferBatch(batch, calcProbability=True)

    # ✅ Spaces are preserved — sentences are returned as-is
    print(f"Recognised : '{recognized[0]}'")
    print(f"Probability: {probability[0]:.4f}")
    return recognized[0]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_char_list() -> str:
    if not os.path.exists(FilePaths.fnCharList):
        raise FileNotFoundError(
            f"charList.txt not found at {FilePaths.fnCharList}.\n"
            f"Run:  python build_charlist.py")
    with codecs.open(FilePaths.fnCharList, encoding="utf-8") as f:
        return f.read()


def choose_decoder(args) -> int:
    if args.beamsearch:
        return DecoderType.BeamSearch
    if args.wordbeamsearch:
        return DecoderType.WordBeamSearch
    return DecoderType.BestPath


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Greek sentence-level HTR — CRNN+CTC")
    parser.add_argument("--train",         action="store_true",
                        help="Train the model")
    parser.add_argument("--validate",      action="store_true",
                        help="Validate the model")
    parser.add_argument("--infer",         action="store_true",
                        help="Run inference on a single image")
    parser.add_argument("--image",         type=str, default=None,
                        help="Path to image for --infer mode")
    parser.add_argument("--beamsearch",    action="store_true",
                        help="Use beam-search decoder")
    parser.add_argument("--wordbeamsearch",action="store_true",
                        help="Use word beam-search decoder")
    args = parser.parse_args()

    decoderType = choose_decoder(args)
    charList    = load_char_list()

    os.makedirs("model_greek/", exist_ok=True)

    # ── train or validate mode ────────────────────────────────────────────────
    if args.train or args.validate:
        loader = DataLoader(
            hf_dataset_name  = HF_DATASET,
            batchSize        = Model.batchSize,
            imgSize          = Model.imgSize,
            maxTextLen       = Model.maxTextLen,
            charList         = charList,
            dataAugmentation = args.train,   # augment only during training
        )

        # Save corpus for optional WordBeamSearch
        corpus_words = loader.trainWords + loader.validationWords
        with open(FilePaths.fnCorpus, "w", encoding="utf-8") as f:
            f.write(" ".join(corpus_words))

        if args.train:
            model = Model(charList, decoderType,
                          mustRestore=False, lastEpoch=0)
            train(model, loader)

        elif args.validate:
            model = Model(charList, decoderType,
                          mustRestore=True)
            validate(model, loader)

    # ── single-image inference mode ───────────────────────────────────────────
    elif args.infer:
        if not args.image:
            print("Error: --infer requires --image <path>")
            sys.exit(1)
        model = Model(charList, decoderType, mustRestore=True)
        infer(model, args.image)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
