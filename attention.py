#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
import math

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))

def attention(inputs, attention_size, time_major=False):

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

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

    return output
