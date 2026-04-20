# -*- coding: utf-8 -*-
"""
main_hindi.py  —  Hindi OCR (Printed Text) - COLAB OPTIMIZED
-------------------------------------------------------------
Entry point for training, validation, and inference on Hindi dataset.

OPTIMIZATIONS FOR FREE-TIER COLAB:
  ✅ Memory monitoring
  ✅ Reduced validation frequency
  ✅ Incremental validation (process in chunks)
  ✅ Automatic cleanup between epochs
  ✅ Resume training from crashes
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import codecs
import gc
import psutil

import cv2
import numpy as np
import tensorflow as tf
import editdistance

from DataLoader_Hindi import DataLoader, Batch
from Model_Hindi import Model, DecoderType
from SamplePreprocessor_Hindi import preprocess

tf.compat.v1.disable_eager_execution()

# ── Paths ─────────────────────────────────────────────────────────────────────
class FilePaths:
    fnCharList  = "model_hindi/charList.txt"
    fnAccuracy  = "model_hindi/accuracy.txt"
    fnCorpus    = "model_hindi/corpus.txt"


# ── Memory monitoring ─────────────────────────────────────────────────────────
def get_memory_usage():
    """Return current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_status(label=""):
    """Print current memory usage."""
    mem_mb = get_memory_usage()
    print(f"[Memory] {label}: {mem_mb:.1f} MB")
    
    # Warning if approaching Colab limit (12GB)
    if mem_mb > 10000:
        print(f"  ⚠️  WARNING: High memory usage ({mem_mb:.1f} MB / 12288 MB limit)")


# ── Learning rate schedule ────────────────────────────────────────────────────
def get_learning_rate(epoch):
    """Learning rate schedule for Hindi printed text."""
    if epoch < 5:
        return 0.001
    elif epoch < 15:
        return 0.0005
    elif epoch < 30:
        return 0.0001
    else:
        return 0.00005


# ── Training ──────────────────────────────────────────────────────────────────
def train(model: Model, loader: DataLoader):
    """Train the model with early stopping on validation CER."""
    epoch              = model.lastEpoch
    bestCharErrorRate  = float("inf")
    noImprovementSince = 0
    earlyStopping      = 10
    
    # ✅ Validate every N epochs (reduce memory pressure on free tier)
    VALIDATION_INTERVAL = 2

    while True:
        epoch += 1
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}")
        print(f"{'='*80}")
        
        print_memory_status("Start of epoch")

        lr = get_learning_rate(epoch)
        print(f"Learning rate: {lr}")

        # ── Train one epoch ───────────────────────────────────────────────────
        print("Training...")
        loader.trainSet()
        epoch_loss = []
        
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch    = loader.getNext()
            loss     = model.trainBatch(batch, learning_rate=lr)
            epoch_loss.append(loss)
            
            # Print progress every 10 batches
            if iterInfo[0] % 10 == 0 or iterInfo[0] == iterInfo[1]:
                print(f"  Batch {iterInfo[0]:>4}/{iterInfo[1]}  loss={loss:.4f}")
            
            # ✅ Memory check during training
            if iterInfo[0] % 50 == 0:
                print_memory_status(f"  After batch {iterInfo[0]}")

        mean_loss = np.mean(epoch_loss)
        print(f"  Mean epoch loss: {mean_loss:.4f}")
        
        # ✅ Cleanup after training
        loader.cleanup()
        gc.collect()
        print_memory_status("After training cleanup")

        # ── Validate (every N epochs) ────────────────────────────────────────
        should_validate = (epoch % VALIDATION_INTERVAL == 0) or (epoch <= 5)
        
        if should_validate:
            charErrorRate, metrics = validate_with_metrics(model, loader)
            
            # Checkpoint logic
            if charErrorRate < bestCharErrorRate:
                print(f"✅ CER improved {bestCharErrorRate*100:.2f}% → "
                      f"{charErrorRate*100:.2f}%  — saving model")
                bestCharErrorRate  = charErrorRate
                noImprovementSince = 0
                model.save(epoch)
                
                with open(FilePaths.fnAccuracy, "w", encoding="utf-8") as f:
                    f.write(f"Epoch {epoch}\n")
                    f.write(f"CER: {charErrorRate*100:.2f}%\n")
                    f.write(f"Loss: {mean_loss:.4f}\n")
                    f.write(f"Space Recall: {metrics['space_recall']:.2%}\n")
                    f.write(f"Space Precision: {metrics['space_precision']:.2%}\n")
            else:
                print(f"  CER did not improve ({charErrorRate*100:.2f}% vs "
                      f"best {bestCharErrorRate*100:.2f}%)")
                noImprovementSince += VALIDATION_INTERVAL
        else:
            print(f"  Skipping validation (will validate at epoch {epoch + (VALIDATION_INTERVAL - epoch % VALIDATION_INTERVAL)})")
            noImprovementSince += 1
            
            # Save checkpoint anyway
            model.save(epoch)

        # ✅ Memory cleanup after validation
        gc.collect()
        print_memory_status("End of epoch")

        if noImprovementSince >= earlyStopping:
            print(f"\nNo improvement for {earlyStopping} epochs — "
                  f"training stopped.")
            break


# ── Validation with metrics (incremental) ────────────────────────────────────
def validate_with_metrics(model: Model, loader: DataLoader):
    """
    Run validation set; return CER and detailed metrics.
    
    ✅ OPTIMIZATION: Process validation in chunks to avoid memory spike.
    """
    print("Validating...")
    loader.validationSet()

    numCharErr   = 0
    numCharTotal = 0
    numWordOK    = 0
    numWordTotal = 0
    
    all_gt_texts = []
    all_pred_texts = []
    
    # ✅ Process in chunks to avoid memory spike
    batch_count = 0
    PRINT_EVERY = 20  # Print less frequently

    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        batch    = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        for i in range(len(recognized)):
            gt   = batch.gtTexts[i]
            pred = recognized[i]
            
            all_gt_texts.append(gt)
            all_pred_texts.append(pred)

            numWordOK    += 1 if gt == pred else 0
            numWordTotal += 1

            dist          = editdistance.eval(pred, gt)
            numCharErr   += dist
            numCharTotal += len(gt)

            status = "[OK]" if dist == 0 else f"[ERR:{dist}]"
            
            # ✅ Print less frequently to reduce console spam
            if batch_count % PRINT_EVERY == 0:
                gt_spaces = gt.count(' ')
                pred_spaces = pred.count(' ')
                
                print(f"  {status}  GT: '{gt[:60]}'")
                print(f"         PRD: '{pred[:60]}'")
                if gt_spaces != pred_spaces:
                    print(f"         ⚠️  Spaces: GT={gt_spaces} PRD={pred_spaces}")
        
        batch_count += 1
        
        # ✅ Cleanup every 50 batches
        if batch_count % 50 == 0:
            gc.collect()

    charErrorRate = numCharErr  / max(numCharTotal, 1)
    wordAccuracy  = numWordOK   / max(numWordTotal, 1)
    
    # Calculate space metrics
    total_gt_spaces = sum(text.count(' ') for text in all_gt_texts)
    total_pred_spaces = sum(text.count(' ') for text in all_pred_texts)
    
    correct_spaces = 0
    for gt, pred in zip(all_gt_texts, all_pred_texts):
        for i in range(min(len(gt), len(pred))):
            if gt[i] == ' ' and pred[i] == ' ':
                correct_spaces += 1
    
    space_recall = correct_spaces / max(total_gt_spaces, 1)
    space_precision = correct_spaces / max(total_pred_spaces, 1)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Character Error Rate: {charErrorRate*100:.2f}%")
    print(f"Word Accuracy:        {wordAccuracy*100:.2f}%")
    print(f"\n📊 Space Statistics:")
    print(f"  Spaces in GT:       {total_gt_spaces}")
    print(f"  Spaces predicted:   {total_pred_spaces}")
    print(f"  Correct spaces:     {correct_spaces}")
    print(f"  Space Recall:       {space_recall:.2%}")
    print(f"  Space Precision:    {space_precision:.2%}")
    
    if total_pred_spaces == 0:
        print("  ⚠️  WARNING: No spaces predicted!")
    
    metrics = {
        'space_recall': space_recall,
        'space_precision': space_precision,
        'word_accuracy': wordAccuracy
    }
    
    # ✅ Cleanup
    loader.cleanup()
    gc.collect()
    
    return charErrorRate, metrics


# ── Validation (simple version) ──────────────────────────────────────────────
def validate(model: Model, loader: DataLoader) -> float:
    """Run validation set; return Character Error Rate."""
    cer, _ = validate_with_metrics(model, loader)
    return cer


# ── Single-image inference ───────────────────────────────────────────────────
def infer(model: Model, img_path: str) -> str:
    """Recognise text in a single image file."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    processed = preprocess(img, Model.imgSize, dataAugmentation=False)
    batch     = Batch(imgs=[processed], gtTexts=None)

    (recognized, probability) = model.inferBatch(batch, calcProbability=True)

    print(f"Recognised : '{recognized[0]}'")
    print(f"Probability: {probability[0]:.4f}")
    print(f"Spaces:      {recognized[0].count(' ')}")
    
    return recognized[0]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_char_list() -> str:
    if not os.path.exists(FilePaths.fnCharList):
        raise FileNotFoundError(
            f"charList.txt not found at {FilePaths.fnCharList}.\n"
            f"Run:  python build_charlist_hindi.py --csv /path/to/data.csv")
    with codecs.open(FilePaths.fnCharList, encoding="utf-8") as f:
        return f.read()


def choose_decoder(args) -> int:
    if args.beamsearch:
        return DecoderType.BeamSearch
    return DecoderType.BestPath


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hindi OCR — CRNN+CTC for printed text (Colab-optimized)")
    
    # Mode
    parser.add_argument("--train",     action="store_true",
                        help="Train the model")
    parser.add_argument("--validate",  action="store_true",
                        help="Validate the model")
    parser.add_argument("--infer",     action="store_true",
                        help="Run inference on a single image")
    
    # Data
    parser.add_argument("--csv",       type=str, default=None,
                        help="Path to data.csv file (required for train/validate)")
    parser.add_argument("--images",    type=str, default=None,
                        help="Path to images directory (required for train/validate)")
    parser.add_argument("--image",     type=str, default=None,
                        help="Path to single image for --infer mode")
    
    # ✅ NEW: Testing mode for free tier
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size for testing (e.g., 1000)")
    
    # Decoder
    parser.add_argument("--beamsearch", action="store_true",
                        help="Use beam-search decoder")
    
    args = parser.parse_args()

    # ✅ Initial memory check
    print_memory_status("Initial")

    decoderType = choose_decoder(args)
    charList    = load_char_list()
    
    # Verify character list
    print(f"\nCharacter list verification:")
    print(f"  Length: {len(charList)}")
    print(f"  Index 0 (CTC blank): {repr(charList[0])}")
    print(f"  Index 1 (space):     {repr(charList[1])}")
    
    if charList[0] != '-':
        print("  ❌ ERROR: Index 0 must be '-' (CTC blank)")
        sys.exit(1)
    if charList[1] != ' ':
        print("  ❌ ERROR: Index 1 must be ' ' (space)")
        sys.exit(1)
    print("  ✅ Character list format is correct")

    os.makedirs("model_hindi/", exist_ok=True)

    # ── Train or validate mode ────────────────────────────────────────────────
    if args.train or args.validate:
        if not args.csv or not args.images:
            print("Error: --train and --validate require --csv and --images")
            sys.exit(1)
        
        # ✅ Show warning if limiting samples
        if args.max_samples:
            print(f"\n⚠️  TESTING MODE: Limited to {args.max_samples} samples")
            print(f"   Remove --max-samples for full training\n")
        
        loader = DataLoader(
            csv_path         = args.csv,
            images_dir       = args.images,
            batchSize        = Model.batchSize,
            imgSize          = Model.imgSize,
            maxTextLen       = Model.maxTextLen,
            charList         = charList,
            dataAugmentation = args.train,
            val_split        = 0.1,
            max_samples      = args.max_samples
        )
        
        print_memory_status("After DataLoader init")

        # Save corpus
        corpus_words = loader.trainWords + loader.validationWords
        with open(FilePaths.fnCorpus, "w", encoding="utf-8") as f:
            f.write(" ".join(corpus_words))

        if args.train:
            model = Model(charList, decoderType,
                          mustRestore=False, lastEpoch=0)
            print_memory_status("After Model init")
            
            try:
                train(model, loader)
            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user")
                print(f"   Last completed epoch: {model.lastEpoch}")
                print(f"   Resume with same command (model auto-restores)")
            except Exception as e:
                print(f"\n\n❌ Training crashed: {e}")
                print(f"   Last completed epoch: {model.lastEpoch}")
                print(f"   Try reducing --max-samples or check memory")
                raise

        elif args.validate:
            model = Model(charList, decoderType,
                          mustRestore=True)
            print_memory_status("After Model init")
            validate(model, loader)

    # ── Single-image inference mode ───────────────────────────────────────────
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
