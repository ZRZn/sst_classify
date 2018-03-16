#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf

def attentionMulti(inputs, attention_size, s, BATCH_SIZE, sen_len, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    # Pos
    W_pos = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b_pos = tf.Variable(tf.truncated_normal([attention_size], mean=0.128, stddev=0.1))
    u_pos = tf.Variable(tf.truncated_normal([attention_size], mean=0.0, stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_pos = tf.tanh(tf.tensordot(inputs, W_pos, axes=1) + b_pos)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_pos = tf.tensordot(v_pos, u_pos, axes=1)  # (B,T) shape



    # # meg
    W_med = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_med = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_med = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_med = tf.tanh(tf.tensordot(inputs, W_med, axes=1) + b_med)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_med = tf.tensordot(v_med, u_med, axes=1)  # (B,T) shape



    # neg
    W_neg = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b_neg = tf.Variable(tf.truncated_normal([attention_size], mean=0.128, stddev=0.1))
    u_neg = tf.Variable(tf.truncated_normal([attention_size], mean=0.0, stddev=0.1))


    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # v_neg = tf.tanh(tf.tensordot(inputs, W_neg, axes=1) + b_neg)
    # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # vu_neg = tf.tensordot(v_neg, u_neg, axes=1)  # (B,T) shape
    #
    s = tf.cast(s, tf.bool)
    # vu = vu_pos * s[:, :, 2] + vu_med * s[:, :, 1] + vu_neg * s[:, :, 0]


    # W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    # b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    # v = tf.tanh(tf.tensordot(inputs, W, axes=1) + b)
    # t = tf.constant(0)
    # def cond_out(t, vu_final):
    #     return t < BATCH_SIZE
    # def body_out(t, vu_final):
    #     i = tf.constant(0)
    #     def conded(i, vus):
    #         return i < sen_len
    #     def body(i, vus):
    #         def getAttention(flag):
    #             if flag == 0:
    #                 # v = tf.tanh(tf.tensordot(inputs[t, i, :], W_neg, axes=1) + b_neg)
    #                 vu = tf.tensordot(v[t, i, :], u_neg, axes=1)
    #             elif flag == 1:
    #                 # v = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b_med)
    #                 vu = tf.tensordot(v[t, i, :], u_med, axes=1)
    #             else:
    #                 # v = tf.tanh(tf.tensordot(inputs[t, i, :], W_pos, axes=1) + b_pos)
    #                 vu = tf.tensordot(v[t, i, :], u_pos, axes=1)
    #             return vu
    #
    #         vu = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
    #                                                                           lambda: getAttention(2)))
    #         vus = tf.concat((vus, [vu]), axis=0)
    #         i += 1
    #         return i, vus
    #
    #     i, vuss = tf.while_loop(conded, body, (i, tf.constant([])), shape_invariants=(i.get_shape(), tf.TensorShape([None])))
    #     vu_final = tf.concat((vu_final, [vuss]), axis=0)
    #     t += 1
    #     return t, vu_final
    #
    # zero = tf.Variable(0, dtype=tf.int32)
    # vu_final = tf.zeros((zero, sen_len))
    # t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))


    # w,u不一样
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
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
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W_neg, axes=1) + b)
                    vu = tf.tensordot(v, u_neg, axes=1)
                elif flag == 1:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
                    vu = tf.tensordot(v, u_med, axes=1)
                else:
                    v = tf.tanh(tf.tensordot(inputs[t, i, :], W_pos, axes=1) + b)
                    vu = tf.tensordot(v, u_pos, axes=1)
                return vu

            vu = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
                                                                              lambda: getAttention(2)))
            vus = tf.concat((vus, [vu]), axis=0)
            i += 1
            return i, vus

        i, vuss = tf.while_loop(conded, body, (i, tf.constant([])),
                                shape_invariants=(i.get_shape(), tf.TensorShape([None])))
        vu_final = tf.concat((vu_final, [vuss]), axis=0)
        t += 1
        return t, vu_final

    zero = tf.Variable(0, dtype=tf.int32)
    vu_final = tf.zeros((zero, sen_len))
    t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))

    # # w不一样
    # b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    # u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
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
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W_neg, axes=1) + b)
    #             elif flag == 1:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W_med, axes=1) + b)
    #             else:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W_pos, axes=1) + b)
    #             return v
    #
    #         vu = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
    #                                                                           lambda: getAttention(2)))
    #         vus = tf.concat((vus, [vu]), axis=0)
    #         i += 1
    #         return i, vus
    #     zero_v = tf.Variable(0, dtype=tf.int32)
    #     vuss = tf.zeros((zero_v, attention_size))
    #     i, vuss = tf.while_loop(conded, body, (i, vuss))
    #     vu_final = tf.concat((vu_final, [vuss]), axis=0)
    #     t += 1
    #     return t, vu_final
    #
    # zero = tf.Variable(0, dtype=tf.int32)
    # vu_final = tf.zeros((zero, sen_len, attention_size))
    # t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))
    # vu_final = tf.tensordot(vu_final, u, axes=1)


    # # u,b不一样
    # W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
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
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_neg)
    #                 vu = tf.tensordot(v, u_neg, axes=1)
    #             elif flag == 1:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_med)
    #                 vu = tf.tensordot(v, u_med, axes=1)
    #             else:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_pos)
    #                 vu = tf.tensordot(v, u_pos, axes=1)
    #             return vu
    #
    #         vu = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
    #                                                                           lambda: getAttention(2)))
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


    # # b不一样
    # W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    # u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
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
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_neg)
    #             elif flag == 1:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_med)
    #             else:
    #                 v = tf.tanh(tf.tensordot(inputs[t, i, :], W, axes=1) + b_pos)
    #             return v
    #
    #         vu = tf.cond(s[t, i, 0], lambda: getAttention(0), lambda: tf.cond(s[t, i, 1], lambda: getAttention(1),
    #                                                                           lambda: getAttention(2)))
    #         vus = tf.concat((vus, [vu]), axis=0)
    #         i += 1
    #         return i, vus
    #     zero_v = tf.Variable(0, dtype=tf.int32)
    #     vuss = tf.zeros((zero_v, attention_size))
    #     i, vuss = tf.while_loop(conded, body, (i, vuss))
    #     vu_final = tf.concat((vu_final, [vuss]), axis=0)
    #     t += 1
    #     return t, vu_final
    #
    # zero = tf.Variable(0, dtype=tf.int32)
    # vu_final = tf.zeros((zero, sen_len, attention_size))
    # t, vu_final = tf.while_loop(cond_out, body_out, (t, vu_final))
    # vu_final = tf.tensordot(vu_final, u, axes=1)


    alphas = tf.nn.softmax(vu_final)  # (B,T) shape also

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output, alphas
