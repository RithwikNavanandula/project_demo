# -*- coding: utf-8 -*-
"""
Model.py  -  Greek Sentence-Level HTR
TensorFlow 2.x + Keras 3 compatible (Colab 2024+)

BatchNorm implemented with pure TF ops + tf.cond so it works
correctly in TF1 graph mode with a symbolic training tensor.
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
    """Sentence-level CRNN+CTC for Greek polytonic handwriting."""

    batchSize      = 8
    imgSize        = (1024, 64)
    maxTextLen     = 128
    CNN_TIME_STEPS = 256

    def __init__(self, charList, decoderType=DecoderType.BestPath,
                 mustRestore=False, lastEpoch=0):

        self.charList    = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID      = 0
        self.lastEpoch   = lastEpoch

        # placeholders
        self.inputImgs = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, Model.imgSize[0], Model.imgSize[1]),
            name='input_imgs')

        self.isTraining = tf.compat.v1.placeholder_with_default(
            False, shape=(), name='is_training')

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.batchesTrained = 0
        self.learningRate   = tf.compat.v1.placeholder(
            tf.float32, shape=[], name='learning_rate')

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                self.learningRate).minimize(self.loss)

        (self.sess, self.saver) = self.setupTF()

    # ── Manual BatchNorm using pure TF ops ───────────────────────────────────
    # This is the key fix: tf.cond handles the symbolic isTraining tensor
    # correctly inside TF1 graph mode — Keras 3 BN layer cannot do this.
    def _batch_norm(self, x, name):
        channels = x.shape[-1]

        gamma       = tf.Variable(tf.ones([channels]),  trainable=True,  name=f'{name}_gamma')
        beta        = tf.Variable(tf.zeros([channels]), trainable=True,  name=f'{name}_beta')
        moving_mean = tf.Variable(tf.zeros([channels]), trainable=False, name=f'{name}_moving_mean')
        moving_var  = tf.Variable(tf.ones([channels]),  trainable=False, name=f'{name}_moving_var')

        decay = 0.99
        eps   = 1e-5

        def train_mode():
            # compute batch statistics
            mean, var = tf.nn.moments(x, axes=[0, 1, 2])
            # update moving averages
            update_mean = tf.compat.v1.assign(
                moving_mean, moving_mean * decay + mean * (1 - decay))
            update_var  = tf.compat.v1.assign(
                moving_var,  moving_var  * decay + var  * (1 - decay))
            # normalise using batch stats (updates run as side-effects)
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

        def infer_mode():
            # normalise using stored moving averages
            return tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, eps)

        return tf.cond(self.isTraining, train_mode, infer_mode)

    # ── CNN ───────────────────────────────────────────────────────────────────
    def setupCNN(self):
        """
        5-block CNN.
        Input  (batch, W=1024, H=64, 1)
        Output (batch, W=256,  H=1,  512)
        """
        cnnIn4d = tf.expand_dims(self.inputImgs, axis=3)

        kernelVals  = [5, 5, 3, 3, 3]
        featureVals = [1, 64, 128, 256, 256, 512]
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
            conv = self._batch_norm(conv, name=f'bn_{i}')   # pure TF BatchNorm
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(
                relu,
                ksize   = (1, poolVals[i][0],  poolVals[i][1],  1),
                strides = (1, strideVals[i][0], strideVals[i][1], 1),
                padding = 'VALID')

        self.cnnOut4d = pool
        print('[Model] CNN output shape:', pool.get_shape())
        # Expected: (batch, 256, 1, 512)

    # ── RNN ───────────────────────────────────────────────────────────────────
    def setupRNN(self):
        """
        Two stacked BiLSTMs.
        Input  (batch, 256, 512)
        Output (batch, 256, num_classes)
        """
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        numHidden = 512
        lstm1 = Bidirectional(LSTM(numHidden, return_sequences=True), name='bilstm_1')
        lstm2 = Bidirectional(LSTM(numHidden, return_sequences=True), name='bilstm_2')

        x = lstm1(rnnIn3d)
        x = tf.nn.dropout(x, rate=0.3)
        x = lstm2(x)

        dense = Dense(len(self.charList) + 1, name='output_dense')
        self.rnnOut3d = dense(x)
        print('[Model] RNN output shape:', self.rnnOut3d.shape)
        # Expected: (batch, 256, num_classes)

    # ── CTC ───────────────────────────────────────────────────────────────────
    def setupCTC(self):
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])

        self.gtTexts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, shape=[None, 2], name='gt_indices'),
            tf.compat.v1.placeholder(tf.int32, shape=[None],    name='gt_values'),
            tf.compat.v1.placeholder(tf.int64, shape=[2],       name='gt_shape'))

        self.seqLen = tf.compat.v1.placeholder(
            tf.int32, shape=[None], name='seq_len')

        self.loss = tf.reduce_mean(
            tf.compat.v1.nn.ctc_loss(
                labels                            = self.gtTexts,
                inputs                            = self.ctcIn3dTBC,
                sequence_length                   = self.seqLen,
                ctc_merge_repeated                = True,
                ignore_longer_outputs_than_inputs = True))

        self.savedCtcInput = tf.compat.v1.placeholder(
            tf.float32,
            shape=[Model.CNN_TIME_STEPS, None, len(self.charList) + 1],
            name='saved_ctc_input')

        self.lossPerElement = tf.compat.v1.nn.ctc_loss(
            labels             = self.gtTexts,
            inputs             = self.savedCtcInput,
            sequence_length    = self.seqLen,
            ctc_merge_repeated = True)

        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(
                inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                beam_width=100)
        elif self.decoderType == DecoderType.WordBeamSearch:
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
            chars     = str().join(self.charList)
            wordChars = open('model_greek/wordCharList.txt').read().splitlines()[0]
            corpus    = open('model_greek/corpus.txt').read()
            self.decoder = word_beam_search_module.word_beam_search(
                tf.nn.softmax(self.ctcIn3dTBC, axis=2), 50, 'Words', 0.0,
                corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

    # ── TF session ────────────────────────────────────────────────────────────
    def setupTF(self):
        print('Python:     ', sys.version)
        print('TensorFlow: ', tf.__version__)

        sess  = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver(max_to_keep=3)

        modelDir       = 'model_greek/'
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

    # ── helpers ───────────────────────────────────────────────────────────────
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

    # ── train ─────────────────────────────────────────────────────────────────
    def trainBatch(self, batch):
        numBatchElements = len(batch.imgs)
        sparse           = self.toSparse(batch.gtTexts)
        feedDict = {
            self.inputImgs   : batch.imgs,
            self.gtTexts     : sparse,
            self.seqLen      : [Model.CNN_TIME_STEPS] * numBatchElements,
            self.learningRate: 0.00001,
            self.isTraining  : True,
        }
        (_, lossVal) = self.sess.run([self.optimizer, self.loss], feedDict)
        self.batchesTrained += 1
        return lossVal

    # ── infer ─────────────────────────────────────────────────────────────────
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

    # ── save ──────────────────────────────────────────────────────────────────
    def save(self, epoch):
        self.snapID += 1
        self.saver.save(self.sess, 'model_greek/snapshot', global_step=epoch)
        print(f'[Model] Checkpoint saved (epoch {epoch})')
