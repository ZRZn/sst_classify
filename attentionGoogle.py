#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import all_path
import tensorflow as tf
import math
import numpy as np

LAYERS_NUM = 3
HEAD_NUM = 4
EACH_HEAD_SIZE = 50
NUM_EPOCHS = 200
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 200
KEEP_PROB = 0.8
DELTA = 0.5
Y_Class = 5
SEN_CLASS = 3
W1_SIZE = 300
W2_SIZE = 200

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))

def add_and_norm(x, sub_layer_x, num=0):
    with tf.variable_scope("add-and-norm-"+str(num)):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x))

def feed_forward_net(input):
    with tf.variable_scope("feed-forward"):
        output = tf.layers.dense(input, W1_SIZE, activation=tf.nn.relu)
        output = tf.layers.dense(output, W2_SIZE)
        return tf.nn.dropout(output, KEEP_PROB)

def attentionGoogle(inputs, hidden_size, B=32, time_major=False):

    assert EACH_HEAD_SIZE * HEAD_NUM == hidden_size

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])


    q = tf.layers.dense(inputs, EACH_HEAD_SIZE * HEAD_NUM, use_bias=False)
    k = tf.layers.dense(inputs, EACH_HEAD_SIZE * HEAD_NUM, use_bias=False)
    v = tf.layers.dense(inputs, EACH_HEAD_SIZE * HEAD_NUM, use_bias=False)

    def multi_head_transpose(tensor, head_num=HEAD_NUM, each_head_size=EACH_HEAD_SIZE):
        tensor = tf.reshape(tensor, [B, -1, head_num, each_head_size])
        return tf.transpose(tensor, [0, 2, 1, 3])   #shape= [B, HEAD_NUM, SEN_LEN, EACH_HEAD_SIZE]

    qs = multi_head_transpose(q)
    ks = multi_head_transpose(k)
    vs = multi_head_transpose(v)

    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (EACH_HEAD_SIZE**0.5)
    o3 = tf.nn.softmax(o2)
    output = tf.matmul(o3, vs)

    output = tf.transpose(output, [0, 2, 1, 3])     #shape= [B, SEN_LEN, HEAD_NUM, EACH_HEAD_SIZE]

    output = tf.reshape(output, [B, -1, EACH_HEAD_SIZE * HEAD_NUM])

    output = tf.layers.dense(output, ATTENTION_SIZE)

    return tf.nn.dropout(output, KEEP_PROB)

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

input_emd += input_pos

o1 = tf.identity(input_emd)
for i in range(1, LAYERS_NUM + 1):
    with tf.variable_scope("layer-"+str(i)):
        o2 = add_and_norm(o1, attentionGoogle(o1, EMBEDDING_SIZE), 1)
        o3 = add_and_norm(o2, feed_forward_net(o2), 2)
        o1 = tf.identity(o3)

final_out = tf.reduce_sum(o3, axis=1)

full_out = tf.layers.dense(final_out, Y_Class)


#Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=full_out))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# Accuracy metric
predict = tf.argmax(full_out, axis=1, name='predict')
label = tf.argmax(input_y, axis=1, name='label')
equal = tf.equal(predict, label)
accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
def start_train():
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
                if epoch > 12 and b % 10 == 0:
                    # print("accuracy_train" == accuracy_train / (b + 1))
                    # Testin
                    accuracy_test = 0
                    # print("origin_test == ", accuracy_test)
                    test_batches = len(test_X) // BATCH_SIZE
                    result_tag = []
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
                        res_max = result_tag
                    print("accuracy_test == ", accuracy_test)
                    print("epoch = ", epoch, "max == ", max_acc)


        print("max_accuracy == ", max_acc)
        return max_acc, res_max

start_train()