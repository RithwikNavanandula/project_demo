# -*- coding: utf-8 -*-
"""
Model_Hindi_v2.py  —  FAST & BALANCED VERSION
--------------------------------------------------------------
Optimized for speed with balanced regularization.

KEY CHANGES:
  ✅ Larger batch size (16-32) - No gradient accumulation
  ✅ Label smoothing (0.05 instead of 0.1) - Gentler
  ✅ Moderate dropout (0.3) - Not too strong
  ✅ Faster training
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


class Model:
    """CRNN+CTC optimized for fast training."""

    # ✅ LARGER batch size for speed
    batchSize      = 16  # Can increase to 32 if GPU allows
    
    imgSize        = (900, 64)
    maxTextLen     = 128
    CNN_TIME_STEPS = 225

    def __init__(self, charList, decoderType=DecoderType.BestPath,
                 mustRestore=False, lastEpoch=0,restorePath=None):

        self.charList    = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID      = 0
        self.lastEpoch   = lastEpoch
        self.restorePath = restorePath

        self.SPACE_IDX = charList.index(' ') if ' ' in charList else 1

        # Placeholders
        self.inputImgs = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, Model.imgSize[0], Model.imgSize[1]),
            name='input_imgs')

        self.isTraining = tf.compat.v1.placeholder_with_default(
            False, shape=(), name='is_training')

        self.dropout_rate = tf.compat.v1.placeholder_with_default(
            0.0, shape=(), name='dropout_rate')

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.learningRate = tf.compat.v1.placeholder(
            tf.float32, shape=[], name='learning_rate')

        self.setupOptimizer()
        (self.sess, self.saver) = self.setupTF()

    # ── BatchNorm ─────────────────────────────────────────────────────────────
    def _batch_norm(self, x, name):
        """BatchNorm layer."""
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
        """5-block CNN with BatchNorm."""
        cnnIn4d = tf.expand_dims(self.inputImgs, axis=3)

        kernelVals  = [5, 5, 3, 3, 3]
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
        Two stacked BiLSTMs with MODERATE dropout.
        
        ✅ Dropout 0.3 (not 0.5) - Balanced regularization
        """
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        numHidden = 384
        
        lstm1 = Bidirectional(
            LSTM(numHidden, return_sequences=True),
            name='bilstm_1')
        
        lstm2 = Bidirectional(
            LSTM(numHidden, return_sequences=True),
            name='bilstm_2')

        x = lstm1(rnnIn3d)
        # ✅ Moderate dropout (0.3, not 0.5)
        x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = lstm2(x)
        x = tf.nn.dropout(x, rate=self.dropout_rate)

        dense = Dense(len(self.charList) + 1, name='output_dense')
        self.rnnOut3d = dense(x)
        print('[Model] RNN output shape:', self.rnnOut3d.shape)

    # ── CTC ───────────────────────────────────────────────────────────────────
    def setupCTC(self):
        """CTC loss with GENTLE label smoothing."""
        num_classes = len(self.charList) + 1
        init_bias = np.zeros(num_classes, dtype=np.float32)
        init_bias[self.SPACE_IDX] = 1.0

        self.space_logit_bias = tf.Variable(
            initial_value=init_bias,
            trainable=True,
            name='space_logit_bias',
            dtype=tf.float32)

        self.ctcIn3dTBC_raw = tf.transpose(self.rnnOut3d, [1, 0, 2])
        self.ctcIn3dTBC = self.ctcIn3dTBC_raw + self.space_logit_bias

        self.gtTexts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, shape=[None, 2], name='gt_indices'),
            tf.compat.v1.placeholder(tf.int32, shape=[None],    name='gt_values'),
            tf.compat.v1.placeholder(tf.int64, shape=[2],       name='gt_shape'))

        self.seqLen = tf.compat.v1.placeholder(
            tf.int32, shape=[None], name='seq_len')

        # ✅ GENTLE label smoothing (0.05 instead of 0.1)
        self.label_smoothing = 0.05
        
        ctc_loss_per_sample = tf.compat.v1.nn.ctc_loss(
            labels             = self.gtTexts,
            inputs             = self.ctcIn3dTBC,
            sequence_length    = self.seqLen,
            ctc_merge_repeated = True,
            ignore_longer_outputs_than_inputs = True)

        # Space weighting
        SPACE_WEIGHT = 2.0
        batch_size_dyn = tf.shape(ctc_loss_per_sample)[0]
        
        gt_values  = self.gtTexts.values
        gt_indices = self.gtTexts.indices
        batch_idx  = gt_indices[:, 0]
        
        is_space = tf.cast(tf.equal(gt_values, self.SPACE_IDX), tf.float32)

        spaces_per_sample = tf.math.unsorted_segment_sum(
            is_space, batch_idx, num_segments=batch_size_dyn)
        labels_per_sample = tf.math.unsorted_segment_sum(
            tf.ones_like(is_space), batch_idx, num_segments=batch_size_dyn)

        space_fraction  = spaces_per_sample / tf.maximum(labels_per_sample, 1.0)
        sample_weights  = 1.0 + (SPACE_WEIGHT - 1.0) * space_fraction

        # Apply gentle label smoothing
        smoothed_loss = ctc_loss_per_sample * (1.0 - self.label_smoothing) + \
                        self.label_smoothing * tf.ones_like(ctc_loss_per_sample)
        
        self.loss = tf.reduce_mean(smoothed_loss * sample_weights)

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

        # Decoder
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(
                inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs         = self.ctcIn3dTBC,
                sequence_length= self.seqLen,
                beam_width     = 100,
                merge_repeated = False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def setupOptimizer(self):
        """Simple Adam optimizer (no gradient accumulation needed)."""
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate)
        
        # Get gradients
        tvars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        
        # Clip gradients
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        
        # Apply gradients
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = optimizer.apply_gradients(zip(grads, tvars))

    # ── TF session ────────────────────────────────────────────────────────────
    def setupTF(self):
        print('Python:     ', sys.version)
        print('TensorFlow: ', tf.__version__)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        
        sess  = tf.compat.v1.Session(config=config)
        saver = tf.compat.v1.train.Saver(max_to_keep=5)

        modelDir       = 'model_hindi/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)

# 🔥 PRIORITY: custom restore path
        if self.restorePath is not None:
            print('[Model] Restoring from:', self.restorePath)
            saver.restore(sess, self.restorePath)
            self.lastEpoch = int(self.restorePath.split('-')[-1])

        elif latestSnapshot:
            print('[Model] Restoring from:', latestSnapshot)
            saver.restore(sess, latestSnapshot)
            self.lastEpoch = int(latestSnapshot.split('-')[-1])

        elif self.mustRestore:
            raise Exception('No saved model found in: ' + modelDir)

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
        decoded = ctcOutput[0][0]
        for (idx, idx2d) in enumerate(decoded.indices):
            label        = decoded.values[idx]
            batchElement = idx2d[0]
            encodedLabelStrs[batchElement].append(label)
        return [str().join([self.charList[c] for c in labelStr])
                for labelStr in encodedLabelStrs]

    # ── Train ─────────────────────────────────────────────────────────────────
    def trainBatch(self, batch, learning_rate=0.0001):
        """Train on one batch (NO gradient accumulation)."""
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        feedDict = {
            self.inputImgs   : batch.imgs,
            self.gtTexts     : sparse,
            self.seqLen      : [Model.CNN_TIME_STEPS] * numBatchElements,
            self.learningRate: learning_rate,
            self.isTraining  : True,
            self.dropout_rate: 0.3,  # ✅ Moderate dropout
        }
        
        _, lossVal = self.sess.run([self.optimizer, self.loss], feedDict)
        return lossVal

    # ── Infer ─────────────────────────────────────────────────────────────────
    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        """Inference with dropout disabled."""
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {
            self.inputImgs : batch.imgs,
            self.seqLen    : [Model.CNN_TIME_STEPS] * numBatchElements,
            self.isTraining: False,
            self.dropout_rate: 0.0,
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
                self.dropout_rate  : 0.0,
            }
            lossVals = self.sess.run(self.lossPerElement, feedDict)
            probs    = np.exp(-lossVals)

        return (texts, probs)

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(self, epoch):
        self.snapID += 1
        self.saver.save(self.sess, 'model_hindi/snapshot', global_step=epoch)
        print(f'[Model] Checkpoint saved (epoch {epoch})')
