# -*- coding: utf-8 -*-
"""
main_hindi_v2.py  —  FAST TRAINING VERSION
-------------------------------------------------------------
Optimized training with proper learning rate schedule.

KEY IMPROVEMENTS:
  ✅ Warmup + cosine decay learning rate
  ✅ Larger batch size = faster training
  ✅ Less verbose logging
  ✅ Better early stopping
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import codecs
import gc
import json
import time
from datetime import datetime
import math

import cv2
import numpy as np
import tensorflow as tf
import editdistance

from DataLoader_Hindi_v2 import DataLoader, Batch
from Model_Hindi_v2 import Model, DecoderType
from SamplePreprocessor_Hindi_v2 import preprocess

tf.compat.v1.disable_eager_execution()

# ── Paths ─────────────────────────────────────────────────────────────────────
class FilePaths:
    fnCharList  = "model_hindi/charList.txt"
    fnAccuracy  = "model_hindi/accuracy.txt"
    fnMetrics   = "model_hindi/metrics.json"


# ── Learning rate schedule ────────────────────────────────────────────────────
def get_learning_rate(epoch, total_epochs=50):
    """
    Warmup + Cosine decay learning rate.
    
    Better than step decay for convergence.
    """
    warmup_epochs = 3
    base_lr = 0.001
    min_lr = 0.00001
    
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ── Metrics tracking ──────────────────────────────────────────────────────────
class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.metrics = {
            'train_loss': [],
            'val_cer': [],
            'val_word_acc': [],
            'learning_rate': [],
            'epochs': [],
            'epoch_times': []
        }
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.metrics = json.load(f)
            except:
                pass
    
    def add_epoch(self, epoch, train_loss, val_cer, val_word_acc, lr, epoch_time):
        """Add metrics for an epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['val_cer'].append(float(val_cer))
        self.metrics['val_word_acc'].append(float(val_word_acc))
        self.metrics['learning_rate'].append(float(lr))
        self.metrics['epoch_times'].append(float(epoch_time))
        self.save()
    
    def save(self):
        """Save metrics."""
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_best_cer(self):
        """Get best CER."""
        if not self.metrics['val_cer']:
            return float('inf')
        return min(self.metrics['val_cer'])


# ── Training ──────────────────────────────────────────────────────────────────
def train(model: Model, loader: DataLoader, total_epochs=50):
    """Train with early stopping."""
    epoch = model.lastEpoch
    
    metrics_tracker = MetricsTracker(FilePaths.fnMetrics)
    bestCharErrorRate = metrics_tracker.get_best_cer()
    
    noImprovementSince = 0
    earlyStopping = 10

    while epoch < total_epochs:
        epoch += 1
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"{'='*80}")

        lr = get_learning_rate(epoch, total_epochs)
        print(f"Learning rate: {lr:.6f}")

        # ── Train ─────────────────────────────────────────────────────────────
        print("Training...")
        loader.trainSet(subset_size=10000)
        epoch_loss = []
        
        batch_count = 0
        while loader.hasNext():
            batch = loader.getNext()
            loss = model.trainBatch(batch, learning_rate=lr)
            epoch_loss.append(loss)
            batch_count += 1
            
            # Print progress every 50 batches
            if batch_count % 50 == 0:
                iterInfo = loader.getIteratorInfo()
                print(f"  Batch {iterInfo[0]:>4}/{iterInfo[1]}  loss={loss:.4f}")

        mean_loss = np.mean(epoch_loss)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch loss: {mean_loss:.4f}  Time: {epoch_time/60:.1f}min")
        
        loader.cleanup()
        gc.collect()

        # ── Validate ──────────────────────────────────────────────────────────
        charErrorRate, wordAccuracy = validate(model, loader)
        
        # Track metrics
        metrics_tracker.add_epoch(epoch, mean_loss, charErrorRate, 
                                 wordAccuracy, lr, epoch_time)
        
        # Checkpoint
        if charErrorRate < bestCharErrorRate:
            improvement = (bestCharErrorRate - charErrorRate) * 100
            print(f"✅ CER improved by {improvement:.2f}pp → Saving model")
            
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save(epoch)
            
            with open(FilePaths.fnAccuracy, "w", encoding="utf-8") as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"CER: {charErrorRate*100:.2f}%\n")
                f.write(f"Word Acc: {wordAccuracy*100:.2f}%\n")
                f.write(f"Loss: {mean_loss:.4f}\n")
        else:
            print(f"  No improvement ({charErrorRate*100:.2f}% vs {bestCharErrorRate*100:.2f}%)")
            noImprovementSince += 1
            
            # Save every 5 epochs
            if epoch % 5 == 0:
                model.save(epoch)

        gc.collect()

        # Early stopping
        if noImprovementSince >= earlyStopping:
            print(f"\nEarly stopping: No improvement for {earlyStopping} epochs")
            print(f"Best CER: {bestCharErrorRate*100:.2f}%")
            break


# ── Validation ────────────────────────────────────────────────────────────────
def validate(model: Model, loader: DataLoader):
    """Run validation."""
    print("Validating...")
    loader.validationSet()

    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    
    batch_count = 0

    while loader.hasNext():
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        for i in range(len(recognized)):
            gt = batch.gtTexts[i]
            pred = recognized[i]

            numWordOK += 1 if gt == pred else 0
            numWordTotal += 1

            dist = editdistance.eval(pred, gt)
            numCharErr += dist
            numCharTotal += len(gt)

            # Print first few predictions
            if batch_count == 0 and i < 3:
                status = "[✓]" if dist == 0 else f"[✗ {dist}]"
                print(f"  {status} GT:  '{gt[:60]}'")
                print(f"       PRD: '{pred[:60]}'")
        
        batch_count += 1

    charErrorRate = numCharErr / max(numCharTotal, 1)
    wordAccuracy = numWordOK / max(numWordTotal, 1)
    
    print(f"\n{'='*80}")
    print(f"VALIDATION: CER={charErrorRate*100:.2f}%  Word Acc={wordAccuracy*100:.2f}%")
    print(f"{'='*80}")
    
    loader.cleanup()
    gc.collect()
    
    return charErrorRate, wordAccuracy


# ── Inference ─────────────────────────────────────────────────────────────────
def infer(model: Model, img_path: str) -> str:
    """Inference on single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    processed = preprocess(img, Model.imgSize, dataAugmentation=False)
    batch = Batch(imgs=[processed], gtTexts=None)

    (recognized, probability) = model.inferBatch(batch, calcProbability=True)

    print(f"Recognised: '{recognized[0]}'")
    print(f"Probability: {probability[0]:.4f}")
    
    return recognized[0]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_char_list() -> str:
    if not os.path.exists(FilePaths.fnCharList):
        raise FileNotFoundError(f"charList.txt not found at {FilePaths.fnCharList}")
    with codecs.open(FilePaths.fnCharList, encoding="utf-8") as f:
        return f.read()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hindi OCR - Fast Training")
    
    parser.add_argument("--train", action="store_true", help="Train")
    parser.add_argument("--validate", action="store_true", help="Validate")
    parser.add_argument("--infer", action="store_true", help="Infer")
    
    parser.add_argument("--dataset", type=str, 
                        default="rajesh-1902/hindi-ocr-dataset",
                        help="HuggingFace dataset name")
    parser.add_argument("--images", type=str, default="images/",
                        help="Path to images directory")
    parser.add_argument("--image", type=str, default=None,
                        help="Image for inference")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total epochs to train")
    parser.add_argument("--beamsearch", action="store_true",
                        help="Use beam search decoder")
    
    args = parser.parse_args()

    decoderType = DecoderType.BeamSearch if args.beamsearch else DecoderType.BestPath
    charList = load_char_list()
    
    print(f"\nCharacter list: {len(charList)} characters")
    print(f"Index 0: {repr(charList[0])} (blank)")
    print(f"Index 1: {repr(charList[1])} (space)")
    
    os.makedirs("model_hindi/", exist_ok=True)

    if args.train or args.validate:
        loader = DataLoader(
            dataset_name=args.dataset,
            images_dir=args.images,
            batchSize=Model.batchSize,
            imgSize=Model.imgSize,
            maxTextLen=Model.maxTextLen,
            charList=charList,
            dataAugmentation=args.train,
            val_split=0.1
        )

        # Save corpus
        
        if args.train:
            model = Model(charList, decoderType, mustRestore=False, lastEpoch=0)
            try:
                train(model, loader, total_epochs=args.epochs)
            except KeyboardInterrupt:
                print(f"\nTraining interrupted at epoch {model.lastEpoch}")

        elif args.validate:
            model = Model(charList, decoderType, mustRestore=True)
            validate(model, loader)

    elif args.infer:
        if not args.image:
            print("Error: --infer requires --image")
            sys.exit(1)
        model = Model(charList, decoderType, mustRestore=True)
        infer(model, args.image)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
