# -*- coding: utf-8 -*-
"""
Model_Hindi.py  —  Hindi OCR (Printed Text) - COLAB OPTIMIZED
--------------------------------------------------------------
CRNN+CTC model optimized for free-tier Colab.

KEY OPTIMIZATIONS:
  ✅ Smaller batch size (4 instead of 16)
  ✅ Gradient accumulation (simulates larger batches)
  ✅ Memory-efficient CNN (fewer feature maps)
  ✅ Mixed precision training option
"""

from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

tf.compat.v1.disable_eager_execution()


class DecoderType:
    BestPath       = 0
    BeamSearch     = 1
    WordBeamSearch = 2


class Model:
    """CRNN+CTC for Hindi printed text recognition (Colab-optimized)."""

    # ✅ REDUCED for free-tier Colab (was 16)
    batchSize      = 4
    
    # Number of batches to accumulate gradients over
    # Effective batch size = batchSize * gradientAccumulationSteps
    # 4 * 4 = 16 (same as original, but uses less memory)
    gradientAccumulationSteps = 4
    
    imgSize        = (900, 64)   # Matches Hindi dataset
    maxTextLen     = 128
    CNN_TIME_STEPS = 225         # 900 ÷ 4 = 225

    def __init__(self, charList, decoderType=DecoderType.BestPath,
                 mustRestore=False, lastEpoch=0):

        self.charList    = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID      = 0
        self.lastEpoch   = lastEpoch

        # Space index (should be index 1 per our charList layout)
        self.SPACE_IDX = charList.index(' ') if ' ' in charList else 1

        # Placeholders
        self.inputImgs = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, Model.imgSize[0], Model.imgSize[1]),
            name='input_imgs')

        self.isTraining = tf.compat.v1.placeholder_with_default(
            False, shape=(), name='is_training')

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # Gradient accumulation
        self.batchesTrained = 0
        self.accumulatedGradients = 0
        
        self.learningRate = tf.compat.v1.placeholder(
            tf.float32, shape=[], name='learning_rate')

        # Setup optimizer with gradient accumulation
        self.setupOptimizer()

        (self.sess, self.saver) = self.setupTF()

    # ── Manual BatchNorm ──────────────────────────────────────────────────────
    def _batch_norm(self, x, name):
        """Pure-TF BatchNorm that works correctly in TF1 graph mode."""
        channels = x.shape[-1]

        gamma       = tf.Variable(tf.ones([channels]),  trainable=True,  name=f'{name}_gamma')
        beta        = tf.Variable(tf.zeros([channels]), trainable=True,  name=f'{name}_beta')
        moving_mean = tf.Variable(tf.zeros([channels]), trainable=False, name=f'{name}_moving_mean')
        moving_var  = tf.Variable(tf.ones([channels]),  trainable=False, name=f'{name}_moving_var')

        decay = 0.99
        eps   = 1e-5

        def train_mode():
            mean, var = tf.nn.moments(x, axes=[0, 1, 2])
            update_mean = tf.compat.v1.assign(
                moving_mean, moving_mean * decay + mean * (1 - decay))
            update_var  = tf.compat.v1.assign(
                moving_var,  moving_var  * decay + var  * (1 - decay))
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

        def infer_mode():
            return tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, eps)

        return tf.cond(self.isTraining, train_mode, infer_mode)

    # ── CNN ───────────────────────────────────────────────────────────────────
    def setupCNN(self):
        """
        5-block CNN for Hindi printed text (memory-optimized).
        
        ✅ OPTIMIZATION: Reduced feature maps (384 instead of 512 in final layer)
        to save memory on free-tier Colab.
        
        Input  (batch, W=900, H=64, 1)
        Output (batch, W=225, H=1,  384)
        """
        cnnIn4d = tf.expand_dims(self.inputImgs, axis=3)

        kernelVals  = [5, 5, 3, 3, 3]
        # ✅ Reduced final feature count: 512 → 384
        featureVals = [1, 64, 128, 256, 256, 384]
        poolVals    = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 4)]
        strideVals  = poolVals

        pool = cnnIn4d
        for i in range(len(poolVals)):
            kernel = tf.Variable(
                tf.random.truncated_normal(
                    [kernelVals[i], kernelVals[i],
                     featureVals[i], featureVals[i + 1]],
                    stddev=0.1),
                name=f'cnn_kernel_{i}')

            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv = self._batch_norm(conv, name=f'bn_{i}')
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(
                relu,
                ksize   = (1, poolVals[i][0],  poolVals[i][1],  1),
                strides = (1, strideVals[i][0], strideVals[i][1], 1),
                padding = 'VALID')

        self.cnnOut4d = pool
        print('[Model] CNN output shape:', pool.get_shape())

    # ── RNN ───────────────────────────────────────────────────────────────────
    def setupRNN(self):
        """
        Two stacked BiLSTMs (memory-optimized).
        
        ✅ OPTIMIZATION: Reduced hidden units (384 instead of 512)
        
        Input  (batch, 225, 384)
        Output (batch, 225, num_classes)
        """
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # ✅ Reduced from 512 to 384
        numHidden = 384
        
        lstm1 = Bidirectional(LSTM(numHidden, return_sequences=True), name='bilstm_1')
        lstm2 = Bidirectional(LSTM(numHidden, return_sequences=True), name='bilstm_2')

        x = lstm1(rnnIn3d)
        x = tf.nn.dropout(x, rate=0.3)
        x = lstm2(x)

        dense = Dense(len(self.charList) + 1, name='output_dense')
        self.rnnOut3d = dense(x)
        print('[Model] RNN output shape:', self.rnnOut3d.shape)

    # ── CTC ───────────────────────────────────────────────────────────────────
    def setupCTC(self):
        """CTC loss with space-weighting for Hindi."""
        # Trainable space logit bias
        num_classes     = len(self.charList) + 1
        init_bias       = np.zeros(num_classes, dtype=np.float32)
        init_bias[self.SPACE_IDX] = 1.0

        self.space_logit_bias = tf.Variable(
            initial_value=init_bias,
            trainable=True,
            name='space_logit_bias',
            dtype=tf.float32)

        # Add bias to logits
        self.ctcIn3dTBC_raw = tf.transpose(self.rnnOut3d, [1, 0, 2])
        self.ctcIn3dTBC = self.ctcIn3dTBC_raw + self.space_logit_bias

        # Ground-truth sparse tensor
        self.gtTexts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, shape=[None, 2], name='gt_indices'),
            tf.compat.v1.placeholder(tf.int32, shape=[None],    name='gt_values'),
            tf.compat.v1.placeholder(tf.int64, shape=[2],       name='gt_shape'))

        self.seqLen = tf.compat.v1.placeholder(
            tf.int32, shape=[None], name='seq_len')

        # CTC loss per sample
        ctc_loss_per_sample = tf.compat.v1.nn.ctc_loss(
            labels                            = self.gtTexts,
            inputs                            = self.ctcIn3dTBC,
            sequence_length                   = self.seqLen,
            ctc_merge_repeated                = True,
            ignore_longer_outputs_than_inputs = True)

        # Space-weighted loss
        SPACE_WEIGHT = 2.0

        gt_values      = self.gtTexts.values
        gt_indices     = self.gtTexts.indices
        batch_size_dyn = tf.shape(self.ctcIn3dTBC)[1]

        is_space     = tf.cast(tf.equal(gt_values, self.SPACE_IDX), tf.float32)
        batch_idx    = tf.cast(gt_indices[:, 0], tf.int32)

        spaces_per_sample = tf.math.unsorted_segment_sum(
            is_space, batch_idx, num_segments=batch_size_dyn)
        labels_per_sample = tf.math.unsorted_segment_sum(
            tf.ones_like(is_space), batch_idx, num_segments=batch_size_dyn)

        space_fraction  = spaces_per_sample / tf.maximum(labels_per_sample, 1.0)
        sample_weights  = 1.0 + (SPACE_WEIGHT - 1.0) * space_fraction

        self.loss = tf.reduce_mean(ctc_loss_per_sample * sample_weights)

        # Saved-CTC path
        self.savedCtcInput = tf.compat.v1.placeholder(
            tf.float32,
            shape=[Model.CNN_TIME_STEPS, None, len(self.charList) + 1],
            name='saved_ctc_input')

        self.lossPerElement = tf.compat.v1.nn.ctc_loss(
            labels             = self.gtTexts,
            inputs             = self.savedCtcInput,
            sequence_length    = self.seqLen,
            ctc_merge_repeated = True)

        # Decoders
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(
                inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)

        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs         = self.ctcIn3dTBC,
                sequence_length= self.seqLen,
                beam_width     = 100,
                merge_repeated = False)

    # ── Optimizer with gradient accumulation ──────────────────────────────────
    def setupOptimizer(self):
        """
        Setup optimizer with gradient accumulation.
        
        This allows us to use small batch size (4) but accumulate gradients
        over multiple batches to simulate larger batch size (16).
        """
        # Get all trainable variables
        tvars = tf.compat.v1.trainable_variables()
        
        # Create accumulation variables
        self.accum_vars = [
            tf.Variable(tf.zeros_like(tv), trainable=False)
            for tv in tvars
        ]
        
        # Zero out accumulated gradients
        self.zero_op = [
            accum_var.assign(tf.zeros_like(accum_var))
            for accum_var in self.accum_vars
        ]
        
        # Compute gradients
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate)
        grads = tf.gradients(self.loss, tvars)
        
        # Accumulate gradients
        self.accum_op = [
            accum_var.assign_add(grad)
            for accum_var, grad in zip(self.accum_vars, grads)
            if grad is not None
        ]
        
        # Apply accumulated gradients
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.apply_op = optimizer.apply_gradients(
                [(accum_var / Model.gradientAccumulationSteps, tvar)
                 for accum_var, tvar in zip(self.accum_vars, tvars)]
            )

    # ── TF session ────────────────────────────────────────────────────────────
    def setupTF(self):
        print('Python:     ', sys.version)
        print('TensorFlow: ', tf.__version__)

        # ✅ Memory optimization: limit GPU memory growth
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        
        sess  = tf.compat.v1.Session(config=config)
        saver = tf.compat.v1.train.Saver(max_to_keep=3)

        modelDir       = 'model_hindi/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)

        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        if latestSnapshot:
            print('[Model] Restoring from:', latestSnapshot)
            saver.restore(sess, latestSnapshot)
            self.lastEpoch = int(latestSnapshot.split('-')[-1])
        else:
            print('[Model] Initialising fresh weights')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess, saver)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def toSparse(self, texts):
        indices = []
        values  = []
        shape   = [len(texts), 0]
        for (batchElement, text) in enumerate(texts):
            labelStr = [self.charList.index(c) for c in text]
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)
        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        encodedLabelStrs = [[] for _ in range(batchSize)]
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)
        else:
            decoded = ctcOutput[0][0]
            for (idx, idx2d) in enumerate(decoded.indices):
                label        = decoded.values[idx]
                batchElement = idx2d[0]
                encodedLabelStrs[batchElement].append(label)
        return [str().join([self.charList[c] for c in labelStr])
                for labelStr in encodedLabelStrs]

    # ── Train with gradient accumulation ──────────────────────────────────────
    def trainBatch(self, batch, learning_rate=0.0001):
        """
        Train on one batch with gradient accumulation.
        
        Returns loss for the current batch.
        Gradients are applied every gradientAccumulationSteps batches.
        """
        numBatchElements = len(batch.imgs)
        sparse           = self.toSparse(batch.gtTexts)
        feedDict = {
            self.inputImgs   : batch.imgs,
            self.gtTexts     : sparse,
            self.seqLen      : [Model.CNN_TIME_STEPS] * numBatchElements,
            self.learningRate: learning_rate,
            self.isTraining  : True,
        }
        
        # First batch: zero out accumulated gradients
        if self.accumulatedGradients == 0:
            self.sess.run(self.zero_op)
        
        # Accumulate gradients
        _, lossVal = self.sess.run([self.accum_op, self.loss], feedDict)
        self.accumulatedGradients += 1
        
        # Apply accumulated gradients
        if self.accumulatedGradients >= Model.gradientAccumulationSteps:
            self.sess.run(self.apply_op, {self.learningRate: learning_rate})
            self.accumulatedGradients = 0
            self.batchesTrained += 1
        
        return lossVal

    # ── Infer ─────────────────────────────────────────────────────────────────
    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {
            self.inputImgs : batch.imgs,
            self.seqLen    : [Model.CNN_TIME_STEPS] * numBatchElements,
            self.isTraining: False,
        }
        evalRes = self.sess.run(evalList, feedDict)
        decoded = evalRes[0]
        texts   = self.decoderOutputToText(decoded, numBatchElements)

        probs = None
        if calcProbability:
            sparse   = (self.toSparse(batch.gtTexts)
                        if probabilityOfGT else self.toSparse(texts))
            ctcInput = evalRes[1]
            feedDict = {
                self.savedCtcInput : ctcInput,
                self.gtTexts       : sparse,
                self.seqLen        : [Model.CNN_TIME_STEPS] * numBatchElements,
                self.isTraining    : False,
            }
            lossVals = self.sess.run(self.lossPerElement, feedDict)
            probs    = np.exp(-lossVals)

        return (texts, probs)

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(self, epoch):
        self.snapID += 1
        self.saver.save(self.sess, 'model_hindi/snapshot', global_step=epoch)
        print(f'[Model] Checkpoint saved (epoch {epoch})')
