#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import pickle
import os
from path import *
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import fully_connected
import numpy as np


NUM_EPOCHS = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 50
USR_SIZE = 1310
PRD_SIZE = 1635
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 100
KEEP_PROB = 0.8
DELTA = 0.5


def AttentionLayer(inputs, name):
    # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
    with tf.variable_scope(name):
        # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
        # 因为使用双向GRU，所以其长度为2×hidden_szie
        u_context = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2]), name='u_context')
        # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
        h = layers.fully_connected(inputs, HIDDEN_SIZE * 2, activation_fn=tf.nn.tanh)
        # shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output


def diffGRU(inputs, pos_neg, B, T, name):
    # inputs, shape = (B, T, E)
    # pos_neg, shape = (B, T, 3)

    posGRU = GRUCell(HIDDEN_SIZE)
    negGRU = GRUCell(HIDDEN_SIZE)
    medGRU = GRUCell(HIDDEN_SIZE)
    flag = tf.cast(pos_neg, tf.bool)

    def cond1(i, outputs):
        return i < B

    def body1(i, outputs):
        state = posGRU.zero_state(1, dtype=tf.float32)

        def cond2(in_i, j, h_state, batch_outputs):
            return j < T

        def body2(in_i, j, h_state, batch_outputs):
            temp_input = tf.reshape(inputs[i][j], [1, EMBEDDING_SIZE])
            temp_out, h_state = tf.cond(flag[i][j][0], lambda: posGRU(temp_input, state), lambda: tf.cond(flag[i][j][1],
                                                                                                        lambda: medGRU(
                                                                                                            temp_input,
                                                                                                            h_state),
                                                                                                        lambda: negGRU(
                                                                                                            temp_input,
                                                                                                            h_state)))
            batch_outputs.append(tf.reshape(temp_out, [EMBEDDING_SIZE]))
            j += 1
            return in_i, j, h_state, batch_outputs

        _, _, _, batch_output = tf.while_loop(cond2, body2, (i, 0, state, []))
        batch_outputs = tf.concat(batch_output, axis=0)
        outputs.append(batch_outputs)
        i += 1
        return i, outputs

    _, output = tf.while_loop(cond1, body1, (0, []))
    output = tf.concat(output, axis=0)
    return output
