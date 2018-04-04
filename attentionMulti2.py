#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
import sys
import math
Y_CLASS = 5

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))

def attentionMulti2(inputs, attention_size, s, BATCH_SIZE, sen_len, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    W = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size), calFan(hidden_size, attention_size)))

    # Pos
    W_pos = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size), calFan(hidden_size, attention_size)))
    b_pos = tf.Variable(tf.zeros([attention_size]))
    u_pos = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # # meg
    W_med = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size), calFan(hidden_size, attention_size)))
    b_med = tf.Variable(tf.zeros([attention_size]))
    u_med = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # neg
    W_neg = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size), calFan(hidden_size, attention_size)))
    b_neg = tf.Variable(tf.zeros([attention_size]))
    u_neg = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    W_zero = tf.Variable(tf.zeros([hidden_size, attention_size]), trainable=False)
    b_init = []
    for i in range(attention_size):
        b_init.append(-99999999999999.0)
    b_zero = tf.Variable(b_init, trainable=False)
    u_zero = tf.Variable(tf.ones([attention_size]), trainable=False)
    s = tf.cast(s, tf.bool)

    bf = tf.Variable(tf.zeros([Y_CLASS]))
    bf_pos = tf.Variable(tf.zeros([Y_CLASS]))
    bf_neg = tf.Variable(tf.zeros([Y_CLASS]))
    bf_zero = tf.Variable(tf.zeros([Y_CLASS]), trainable=False)
    wf = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf_pos = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf_neg = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf_zero = tf.Variable(tf.zeros([hidden_size, Y_CLASS]), trainable=False)
    # w,u不一样
    b = tf.Variable(tf.zeros([attention_size]))
    u = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))
    t = tf.constant(0)

    def cond_out(t, vu_final, full_final):
        return t < BATCH_SIZE

    def body_out(t, vu_final, full_final):
        i = tf.constant(0)

        def conded(i, vus, fulls):
            return i < sen_len

        def body(i, vus, fulls):
            def getAttention(flag):
                if flag == 0:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b)
                    vu = tf.tensordot(v, u, axes=1)
                    full = tf.tensordot(inputs[t, i, :], wf_neg, axes=1) + bf_neg
                elif flag == 1:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b)
                    vu = tf.tensordot(v, u, axes=1)
                    full = tf.tensordot(inputs[t, i, :], wf, axes=1) + bf
                elif flag == 2:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b)
                    vu = tf.tensordot(v, u, axes=1)
                    full = tf.tensordot(inputs[t, i, :], wf_pos, axes=1) + bf_pos
                else:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W_zero, axes=1) + b_zero)
                    vu = tf.tensordot(v, u_zero, axes=1)
                    full = tf.tensordot(inputs[t, i, :], wf_zero, axes=1) + bf_zero
                return vu, full

            vu, full = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
                                                                              lambda: tf.cond(s[t, i, 2], lambda: getAttention(2),
                                                                              lambda: getAttention(3))))
            vus = tf.concat((vus, [vu]), axis=0)
            fulls = tf.concat((fulls, [full]), axis=0)
            i += 1
            return i, vus, fulls

        zero_v = tf.Variable(0, dtype=tf.int32)
        fulls = tf.zeros((zero_v, Y_CLASS))
        i, vuss, fulls = tf.while_loop(conded, body, (i, tf.constant([]), fulls),
                                shape_invariants=(i.get_shape(), tf.TensorShape([None]), fulls.get_shape()))
        vu_final = tf.concat((vu_final, [vuss]), axis=0)

        full_final = tf.concat((full_final, [fulls]), axis=0)
        t += 1
        return t, vu_final, full_final

    zero = tf.Variable(0, dtype=tf.int32)
    vu_final = tf.zeros((zero, sen_len))
    zero2 = tf.Variable(0, dtype=tf.int32)
    full_final = tf.zeros((zero2, sen_len, Y_CLASS))
    t, vu_final, full_final = tf.while_loop(cond_out, body_out, (t, vu_final, full_final))



    alphas = tf.nn.softmax(vu_final)  # (B,T) shape also

    output = tf.reduce_sum(full_final * tf.expand_dims(alphas, -1), 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output, alphas
