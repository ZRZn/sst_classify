#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
import math
Y_CLASS = 5
def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))

def attentionCopy(inputs, attention_size, s, BATCH_SIZE, sen_len, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    # Trainable parameters
    W_a = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size), calFan(hidden_size, attention_size)))
    b_omega = tf.Variable(tf.zeros([attention_size]))
    u_omega = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(inputs, W_a, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape


    alphas = tf.nn.softmax(vu)  # (B,T) shape also

    # output = inputs * tf.expand_dims(alphas, -1)

    # # w不一样
    b1 = tf.Variable(tf.zeros([Y_CLASS]))
    b2 = tf.Variable(tf.zeros([Y_CLASS]))
    b3 = tf.Variable(tf.zeros([Y_CLASS]))
    b4 = tf.Variable(tf.zeros([Y_CLASS]))
    b5 = tf.Variable(tf.zeros([Y_CLASS]))
    wf1 = tf.Variable(tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf2 = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf3 = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf4 = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))
    wf5 = tf.Variable(
        tf.random_uniform([hidden_size, Y_CLASS], -calFan(hidden_size, Y_CLASS), calFan(hidden_size, Y_CLASS)))

    t = tf.constant(0)
    def cond_out(t, vu_final):
        return t < BATCH_SIZE

    def body_out(t, vu_final):
        i = tf.constant(0)

        def conded(i, vus):
            return i < sen_len

        def body(i, vus):
            def getAttention(flag):
                if flag == 0:
                    v = tf.tensordot(inputs[t, i, :], wf1, axes=1) + b1
                elif flag == 1:
                    v = tf.tensordot(inputs[t, i, :], wf1, axes=1) + b2
                elif flag == 2:
                    v = tf.tensordot(inputs[t, i, :], wf1, axes=1) + b3
                elif flag == 3:
                    v = tf.tensordot(inputs[t, i, :], wf1, axes=1) + b4
                else:
                    v = tf.tensordot(inputs[t, i, :], wf1, axes=1) + b5
                return v

            vu = tf.cond(tf.less(s[t, i], 0.2), lambda: getAttention(0), lambda: tf.cond(tf.less(s[t, i], 0.4), lambda: getAttention(1),
                                                lambda: tf.cond(tf.less_equal(s[t, i], 0.6), lambda: getAttention(2),
                                                lambda:  tf.cond(tf.less_equal(s[t, i], 0.8), lambda: getAttention(3),
                                                                 lambda: getAttention(4)))))
            vus = tf.concat((vus, [vu]), axis=0)
            i += 1
            return i, vus
        zero_v = tf.Variable(0, dtype=tf.int32)
        vuss = tf.zeros((zero_v, Y_CLASS))
        i, vuss = tf.while_loop(conded, body, (i, vuss))
        vu_final = tf.concat((vu_final, [vuss]), axis=0)
        t += 1
        return t, vu_final

    zero = tf.Variable(0, dtype=tf.int32)
    vu_final = tf.zeros((zero, sen_len, Y_CLASS))
    t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))


    output = tf.reduce_sum(vu_final * tf.expand_dims(alphas, -1), 1)

    # output = tf.reduce_sum(vu_final, 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output, b1, b2, b3, b4, b5

