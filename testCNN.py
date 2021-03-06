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


NUM_EPOCHS = 12
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 200
KEEP_PROB = 0.5
DELTA = 0.5
Y_Class = 5
SEN_CLASS = 3
FILTER_START = 3
FILTER_END = 5
FILTER_NUM = 100
FULL_SIZE = (FILTER_END - FILTER_START + 1) * FILTER_NUM
L2_LAMBDA = 3.0



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


#placeholders

input_x = tf.placeholder(tf.int32, [BATCH_SIZE, None])
input_y = tf.placeholder(tf.int32, [BATCH_SIZE, Y_Class])
input_s = tf.placeholder(tf.int32, [BATCH_SIZE, None, SEN_CLASS])
input_f = tf.placeholder(tf.float32, [BATCH_SIZE, None])
sen_len_ph = tf.placeholder(tf.int32)
keep_prob_ph = tf.placeholder(tf.float32)


#Embedding Layer
emd_file = open(all_path + "emb_array.pkl", "rb")
emb_array = pickle.load(emd_file)
emd_file.close()
embeddings = tf.Variable(emb_array, trainable=True)
input_emd = tf.nn.embedding_lookup(embeddings, input_x)     #shape= (B, None, E)

input_cnn = tf.expand_dims(input_emd, -1)

cnn_out = []

for t in range(FILTER_START, FILTER_END + 1):
    with tf.name_scope("conv-maxpool-" + str(t)):
        filter_shape = [t, EMBEDDING_SIZE, 1, FILTER_NUM]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[FILTER_NUM]), name="b")
        conv = tf.nn.conv2d(input_cnn, W, [1, 1, 1, 1], 'VALID', name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')     #shape=[B, sen_len-filter_size+1, 1, FILTER_NUMS]
        h = tf.reshape(h, [BATCH_SIZE, -1, FILTER_NUM])
        # pooled = attention(h, ATTENTION_SIZE)
        pooled = tf.reduce_max(h, axis=1, name='pooled')   #shape=[B,FILTER_NUMS]
        cnn_out.append(pooled)

cnn_out = tf.concat(cnn_out, axis=1)
# cnn_out = tf.reshape(cnn_out, [-1, FULL_SIZE])


#Dropout
drop_out = tf.nn.dropout(cnn_out, keep_prob_ph)

l2_loss = tf.constant(0.0)
#FullConnect Layer
w_full = tf.Variable(tf.random_uniform([FULL_SIZE, Y_Class], -calFan(FULL_SIZE, Y_Class), calFan(FULL_SIZE, Y_Class)))

b_full = tf.Variable(tf.constant(0.1, shape=[Y_Class]))
full_out = tf.nn.xw_plus_b(drop_out, w_full, b_full)

l2_loss += tf.nn.l2_loss(w_full)
l2_loss += tf.nn.l2_loss(b_full)
#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=full_out)) + L2_LAMBDA * l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# Accuracy metric
predict = tf.argmax(full_out, axis=1, name='predict')
label = tf.argmax(input_y, axis=1, name='label')
equal = tf.equal(predict, label)
accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

def start_cnn():
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
                loss_tr, acc, _ = sess.run([loss, accuracy, optimizer],
                                           feed_dict={input_x: x_train,
                                                      input_y: y_train,
                                                      input_s: s_train,
                                                      input_f: f_train,
                                                      sen_len_ph: sen_len,
                                                      keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                if epoch >= 7:
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
                        loss_test_batch, test_acc = sess.run([loss, accuracy],
                                                             feed_dict={input_x: x_test,
                                                                        input_y: y_test,
                                                                        input_s: s_test,
                                                                        input_f: f_test,
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


max_acc = start_cnn()
