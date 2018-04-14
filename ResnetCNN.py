#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import os
from path import *
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.layers import fully_connected
import numpy as np
from attentionMulti import attentionMulti
from attention import attention
from attentionOri import attentionOri
from attentionCopy import attentionCopy
# from sortData import sortData
# from getInput import read_data, read_y
import math

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))


NUM_EPOCHS = 150
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 200
KEEP_PROB = 0.5
DELTA = 0.5
Y_Class = 5
SEN_CLASS = 3
FILTER_SIZE = 3
FILTER_NUM = 400
L2_LAMBDA = 3.0
LAYER_NUM = 3
NHIDS_LIST = [200, 200, 200, 200]
KWIDTHS_LIST = [3, 3, 3, 3]

#Load Data
train_fir = open(all_path + "train_out.pkl", "rb")
test_fir = open(all_path + "test_out.pkl", "rb")
train_X = pickle.load(train_fir)
train_Y = pickle.load(train_fir)
train_S = pickle.load(train_fir)
train_F = pickle.load(train_fir)

test_X = pickle.load(test_fir)
test_Y = pickle.load(test_fir)
test_S = pickle.load(test_fir)
test_F = pickle.load(test_fir)

train_fir.close()
test_fir.close()


def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


def linear_mapping_weightnorm(inputs, out_dim, dropout, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        input_shape_tensor = tf.shape(inputs)
        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                0.9 * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)  # weightnorm bias is init zero

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        # inputs = tf.matmul(inputs, V)    # x*v

        scaler = tf.div(g, tf.norm(V, axis=0))  # g/2-norm(v)
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])  # x*v g/2-norm(v) + b

        return inputs


def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2] / 2)]
    input_gate = inputs[:, :, int(input_shape[2] / 2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)


def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME"):
    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * 0.9 / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs


def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout):
    next_layer = inputs
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout,
                                                   var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(inputs=next_layer, keep_prob=dropout)

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout * 2,
                                       kernel_size=kwidths_list[layer_idx], padding="SAME")
        next_layer = gated_linear_units(next_layer)
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)

    return next_layer

    #placeholders
input_x = tf.placeholder(tf.int32, [BATCH_SIZE, None])
input_y = tf.placeholder(tf.int32, [BATCH_SIZE, Y_Class])
input_s = tf.placeholder(tf.int32, [BATCH_SIZE, None, SEN_CLASS])
input_f = tf.placeholder(tf.float32, [BATCH_SIZE, None])
sen_len_ph = tf.placeholder(tf.int32)
keep_prob_ph = tf.placeholder(tf.float32)
pos_ph = tf.placeholder(tf.int32, [BATCH_SIZE, None])

#Embedding Layer
emd_file = open(all_path + "emb_array.pkl", "rb")
emb_array = pickle.load(emd_file)
emd_file.close()
embeddings = tf.Variable(emb_array, trainable=True)
input_emd = tf.nn.embedding_lookup(embeddings, input_x)     #shape= (B, None, E)

embeddings_pos = tf.Variable(tf.ones([54, EMBEDDING_SIZE]), trainable=True)
input_pos = tf.nn.embedding_lookup(embeddings_pos, pos_ph)

input_emd = tf.add(input_emd, input_pos)

input_emd = tf.nn.dropout(input_emd, keep_prob_ph)

linear_emd = linear_mapping_weightnorm(input_emd, EMBEDDING_SIZE, dropout=keep_prob_ph)

with tf.variable_scope("encoder_cnn"):
    next_layer = linear_emd
    next_layer = conv_encoder_stack(next_layer, NHIDS_LIST, KWIDTHS_LIST, keep_prob_ph)
    next_layer = linear_mapping_weightnorm(next_layer, EMBEDDING_SIZE, dropout=keep_prob_ph)
    cnn_out = (next_layer + linear_emd) * tf.sqrt(0.5)

final_cnn_state = tf.reduce_mean(cnn_out, axis=1)

#FullConnect Layer
w_full = tf.Variable(tf.random_uniform([EMBEDDING_SIZE, Y_Class], -calFan(EMBEDDING_SIZE, Y_Class), calFan(EMBEDDING_SIZE, Y_Class)))

b_full = tf.Variable(tf.constant(0.1, shape=[Y_Class]))
full_out = tf.nn.xw_plus_b(final_cnn_state, w_full, b_full)


#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=full_out))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# Accuracy metric
predict = tf.argmax(full_out, axis=1, name='predict')
label = tf.argmax(input_y, axis=1, name='label')
equal = tf.equal(predict, label)
accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

def start_res_cnn():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        max_acc = 0
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: ", epoch)

            # Training
            num_batches = len(train_X) // BATCH_SIZE

            indices = np.arange(num_batches)
            np.random.shuffle(indices)

            for b in range(num_batches):
                count = indices[b]
                x_train = train_X[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
                y_train = train_Y[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
                s_train = train_S[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
                f_train = train_F[count * BATCH_SIZE: (count + 1) * BATCH_SIZE]
                sen_len = len(x_train[0])
                pos_train = [[pos for pos in range(sen_len)]] * BATCH_SIZE
                loss_tr, acc, _ = sess.run([loss, accuracy, optimizer],
                                           feed_dict={input_x: x_train,
                                                      input_y: y_train,
                                                      input_s: s_train,
                                                      input_f: f_train,
                                                      pos_ph: pos_train,
                                                      sen_len_ph: sen_len,
                                                      keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                if epoch >= 0:
                    # print("accuracy_train" == accuracy_train / (b + 1))
                    # Testin
                    accuracy_test = 0
                    # print("origin_test == ", accuracy_test)
                    test_batches = len(test_X) // BATCH_SIZE
                    for z in range(test_batches):
                        x_test = test_X[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                        y_test = test_Y[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                        s_test = test_S[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                        f_test = test_F[z * BATCH_SIZE: (z + 1) * BATCH_SIZE]
                        test_len = len(x_test[0])
                        pos_test = [[pos for pos in range(test_len)]] * BATCH_SIZE
                        loss_test_batch, test_acc = sess.run([loss, accuracy],
                                                             feed_dict={input_x: x_test,
                                                                        input_y: y_test,
                                                                        input_s: s_test,
                                                                        input_f: f_test,
                                                                        pos_ph: pos_test,
                                                                        sen_len_ph: test_len,
                                                                        keep_prob_ph: 1.0})
                        accuracy_test += test_acc
                        loss_test += loss_test_batch
                    accuracy_test /= test_batches
                    loss_test /= test_batches
                    if accuracy_test > max_acc:
                        max_acc = accuracy_test
                        # print(res_max)
                        # w_max = W.eval()
                    print("accuracy_test == ", accuracy_test)
                    print("epoch = ", epoch, "max == ", max_acc)


        print("max_accuracy == ", max_acc)
        return max_acc


max_acc = start_res_cnn()
