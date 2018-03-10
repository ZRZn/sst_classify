#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
f = open(all_path + "wbu.pkl", "rb")
w_o = pickle.load(f)
b_o = pickle.load(f)
u_o = pickle.load(f)
f.close()

def attentionOri(inputs, attention_size, time_major=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer


    # Trainable parameters
    W_a = tf.Variable(w_o, trainable=False)
    b_omega = tf.Variable(b_o, trainable=False)
    u_omega = tf.Variable(u_o, trainable=False)

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(inputs, W_a, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape


    alphas = tf.nn.softmax(vu)  # (B,T) shape also

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output, alphas