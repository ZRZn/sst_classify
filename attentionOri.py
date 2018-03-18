#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
import math

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))
def attentionOri(emb_input, inputs, attention_size, s, BATCH_SIZE, sen_len, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    b = tf.Variable(tf.zeros([attention_size]))
    u = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # Pos
    W_pos = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size),
                                          calFan(hidden_size, attention_size)))
    b_pos = tf.Variable(tf.truncated_normal([attention_size], mean=0.128, stddev=0.1))
    u_pos = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_pos = tf.tanh(tf.tensordot(inputs, W_pos, axes=1) + b_pos)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_pos = tf.tensordot(v_pos, u_pos, axes=1)  # (B,T) shape



    # # meg
    W_med = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size),
                                          calFan(hidden_size, attention_size)))
    b_med = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_med = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_med = tf.tanh(tf.tensordot(inputs, W_med, axes=1) + b_med)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_med = tf.tensordot(v_med, u_med, axes=1)  # (B,T) shape



    # neg
    W_neg = tf.Variable(tf.random_uniform([hidden_size, attention_size], -calFan(hidden_size, attention_size),
                                          calFan(hidden_size, attention_size)))
    b_neg = tf.Variable(tf.truncated_normal([attention_size], mean=0.128, stddev=0.1))
    u_neg = tf.Variable(tf.random_uniform([attention_size], -calFan(attention_size, 1), calFan(attention_size, 1)))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_neg = tf.tanh(tf.tensordot(inputs, W_neg, axes=1) + b_neg)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_neg = tf.tensordot(v_neg, u_neg, axes=1)  # (B,T) shape


    # vu = vu_pos * s + vu_med * (1 - 2 * tf.abs(0.5 - s)) + vu_neg * (1 - s)


    # # w,u不一样
    # t = tf.constant(0)
    #
    # def cond_out(t, vu_final):
    #     return t < BATCH_SIZE
    #
    # def body_out(t, vu_final):
    #     i = tf.constant(0)
    #
    #     def conded(i, vus):
    #         return i < sen_len
    #
    #     def body(i, vus):
    #         def getAttention(flag):
    #             if flag == 0:
    #                 v1 = tf.tanh(tf.tensordot(inputs[t, i, :], W_neg, axes=1) + b)
    #                 vu1 = tf.tensordot(v1, u_neg, axes=1)
    #                 v2 = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
    #                 vu2 = tf.tensordot(v2, u_med, axes=1)
    #                 vu = 2 * s[t, i] * vu2 + (1 - 2 * s[t, i]) * vu1
    #             elif flag == 1:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
    #                 vu = tf.tensordot(v, u_med, axes=1)
    #             else:
    #                 v1 = tf.tanh(tf.tensordot(inputs[t, i, :], W_pos, axes=1) + b)
    #                 vu1 = tf.tensordot(v1, u_pos, axes=1)
    #                 v2 = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
    #                 vu2 = tf.tensordot(v2, u_med, axes=1)
    #                 vu = (2 * s[t, i] - 1) * vu1 + (2 - 2 * s[t, i]) * vu2
    #             return vu
    #
    #         vu = tf.cond(tf.less(s[t, i], 0.5), lambda: getAttention(0),
    #                      lambda: tf.cond(tf.equal(s[t, i], 0.5), lambda: getAttention(1),
    #                                      lambda: getAttention(2)))
    #         vus = tf.concat((vus, [vu]), axis=0)
    #         i += 1
    #         return i, vus
    #
    #     i, vuss = tf.while_loop(conded, body, (i, tf.constant([])),
    #                             shape_invariants=(i.get_shape(), tf.TensorShape([None])))
    #     vu_final = tf.concat((vu_final, [vuss]), axis=0)
    #     t += 1
    #     return t, vu_final
    #
    # zero = tf.Variable(0, dtype=tf.int32)
    # vu_final = tf.zeros((zero, sen_len))
    # t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))


    # w不一样
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
                    vu = tf.tanh((1 - 4 * tf.square(s[t, i])) * tf.tensordot(emb_input[t, i, :], W_neg, axes=1) +
                                 tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
                elif flag == 1:
                    vu = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
                else:
                    vu = tf.tanh((1 - 4 * tf.square(s[t, i] - 1)) * tf.tensordot(emb_input[t, i, :], W_pos, axes=1) +
                                 tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
                return vu

            vu = tf.cond(tf.less(s[t, i], 0.5), lambda: getAttention(0),
                         lambda: tf.cond(tf.equal(s[t, i], 0.5), lambda: getAttention(1),
                                         lambda: getAttention(2)))
            vus = tf.concat((vus, [vu]), axis=0)
            i += 1
            return i, vus

        zero_v = tf.Variable(0, dtype=tf.int32)
        vuss = tf.zeros((zero_v, attention_size))
        i, vuss = tf.while_loop(conded, body, (i, vuss))
        vu_final = tf.concat((vu_final, [vuss]), axis=0)
        t += 1
        return t, vu_final

    zero = tf.Variable(0, dtype=tf.int32)
    vu_final = tf.zeros((zero, sen_len, attention_size))
    t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))
    vu_final = tf.tensordot(vu_final, u, axes=1)


    alphas = tf.nn.softmax(vu_final)  # (B,T) shape also

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output, alphas

