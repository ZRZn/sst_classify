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
from attention import attention
import math

def calFan(fan_in, fan_out):
    return math.sqrt(6 / (fan_in + fan_out))

NUM_EPOCHS = 4
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_SIZE = 200
ATTENTION_SIZE = 200
KEEP_PROB = 0.8
DELTA = 0.5
Y_Class = 5
SEN_CLASS = 3



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

state = tf.zeros([1, HIDDEN_SIZE])
state_rev = tf.zeros([1, HIDDEN_SIZE])
posGRU = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)
negGRU = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)
medGRU = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)

posGRU_rev = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)
negGRU_rev = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)
medGRU_rev = GRUCell(HIDDEN_SIZE, reuse=tf.AUTO_REUSE)



def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

def AttentionLayer(inputs, name):
    # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
    with tf.variable_scope(name):
        # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
        # 因为使用双向GRU，所以其长度为2×hidden_szie
        u_context = tf.Variable(tf.truncated_normal([ATTENTION_SIZE]), name='u_context')
        # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
        h = layers.fully_connected(inputs, ATTENTION_SIZE, activation_fn=tf.nn.tanh)
        # shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output


def diffGRU(inputs, pos_neg, B, T, name='RNN'):
    global state
    # inputs, shape = (B, T, E)
    # pos_neg, shape = (B, T, 3)
    t = tf.constant(0)
    with tf.variable_scope(name):
        flag = tf.cast(pos_neg, tf.bool)

        def cond1(i, grus):
            return i < B

        def body1(i, grus):

            # tf.get_variable_scope().reuse_variables()

            def cond2(j, time_outs):
                return j < T

            def body2(j, time_outs):
                global state
                def MultiGRU(temp_value, temp_state, temp_np):
                    tf.get_variable_scope().reuse_variables()
                    if temp_np == 2:
                        return posGRU(temp_value, temp_state)
                    elif temp_np == 1:
                        return medGRU(temp_value, temp_state)
                    else:
                        return negGRU(temp_value, temp_state)

                # temp_input = tf.reshape(inputs[i][j], [1, EMBEDDING_SIZE])
                temp_input =inputs[i][j]
                temp_out, state = tf.cond(flag[i][j][0], lambda: MultiGRU(temp_input, state, 0),
                                            lambda: tf.cond(flag[i][j][1],
                                                            lambda: MultiGRU(
                                                                temp_input,
                                                                state, 1),
                                                            lambda: MultiGRU(
                                                                temp_input,
                                                                state, 2)))

                time_outs = tf.concat((time_outs, temp_out), axis=0)
                j += 1
                return j, time_outs

            j = tf.constant(0)
            # state = posGRU.zero_state(1, dtype=tf.float32)
            zero_time = tf.Variable(0, dtype=tf.int32)
            time_out_zero = tf.zeros((zero_time, HIDDEN_SIZE))
            _, time_outs = tf.while_loop(cond2, body2, (j, time_out_zero))
            grus = tf.concat((grus, [time_outs]), axis=0)
            i += 1
            return i, grus
        zero = tf.Variable(0, dtype=tf.int32)
        grus = tf.zeros((zero, sen_len_ph, HIDDEN_SIZE))
        _, grus = tf.while_loop(cond1, body1, (t, grus))
        return grus

def diffGRURev(inputs, pos_neg, B, T, name='RNN'):
    global state_rev
    # inputs, shape = (B, T, E)
    # pos_neg, shape = (B, T, 3)
    t = tf.constant(0)
    with tf.variable_scope(name):
        flag = tf.cast(pos_neg, tf.bool)

        def cond1(i, grus):
            return i < B

        def body1(i, grus):

            # tf.get_variable_scope().reuse_variables()

            def cond2(j, time_outs):
                return j < T

            def body2(j, time_outs):
                global state_rev
                def MultiGRU(temp_value, temp_state, temp_np):
                    tf.get_variable_scope().reuse_variables()
                    if temp_np == 2:
                        return posGRU_rev(temp_value, temp_state)
                    elif temp_np == 1:
                        return medGRU_rev(temp_value, temp_state)
                    else:
                        return negGRU_rev(temp_value, temp_state)

                # temp_input = tf.reshape(inputs[i][j], [1, EMBEDDING_SIZE])
                temp_input = inputs[i][sen_len_ph - j - 1]
                temp_out, state_rev = tf.cond(flag[i][sen_len_ph - j - 1][0], lambda: MultiGRU(temp_input, state_rev, 0),
                                            lambda: tf.cond(flag[i][sen_len_ph - j - 1][1],
                                                            lambda: MultiGRU(
                                                                temp_input,
                                                                state_rev, 1),
                                                            lambda: MultiGRU(
                                                                temp_input,
                                                                state_rev, 2)))

                time_outs = tf.concat((temp_out, time_outs), axis=0)
                j += 1
                return j, time_outs

            j = tf.constant(0)
            # state = posGRU.zero_state(1, dtype=tf.float32)
            zero_time = tf.Variable(0, dtype=tf.int32)
            time_out_zero = tf.zeros((zero_time, HIDDEN_SIZE))
            _, time_outs = tf.while_loop(cond2, body2, (j, time_out_zero))
            grus = tf.concat((grus, [time_outs]), axis=0)
            i += 1
            return i, grus

        zero = tf.Variable(0, dtype=tf.int32)
        grus = tf.zeros((zero, sen_len_ph, HIDDEN_SIZE))
        _, grus = tf.while_loop(cond1, body1, (t, grus))
        return grus


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

# # #normal bi_GRU
# (f_out, b_out), _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE), input_emd, sequence_length=length(input_emd), dtype=tf.float32)
# gru_out = tf.concat((f_out, b_out), axis=2)

# DIFF-GRU Layer
input_emd = tf.reshape(input_emd, (BATCH_SIZE, sen_len_ph, -1, EMBEDDING_SIZE))
gru_out_f = diffGRU(input_emd, input_s, BATCH_SIZE, sen_len_ph, 'RNN')
gru_out_b = diffGRURev(input_emd, input_s, BATCH_SIZE, sen_len_ph, 'RNN_REV')
gru_out = tf.concat((gru_out_f, gru_out_b), axis=2)
print(gru_out.shape[2].value)


#RNN
# gru_out, _ = dynamic_rnn(BasicRNNCell(HIDDEN_SIZE), input_emd, sequence_length=length(input_emd), dtype=tf.float32)

#Attention Layer
# attention_output, alphas = attentionMulti(gru_out, ATTENTION_SIZE, input_s, BATCH_SIZE, sen_len_ph)

attention_output, w_a, b_omega, u_omega = attention(gru_out, ATTENTION_SIZE)
# attention_output, alphas = attentionOri(input_emd, gru_out, ATTENTION_SIZE, input_f, BATCH_SIZE, sen_len_ph)

#Dropout
drop_out = tf.nn.dropout(attention_output, keep_prob_ph)

#FullConnect Layer
w_full = tf.Variable(tf.random_uniform([gru_out.shape[2].value, Y_Class], -calFan(gru_out.shape[2].value, Y_Class), calFan(gru_out.shape[2].value, Y_Class)))

b_full = tf.Variable(tf.zeros(shape=[Y_Class]))
full_out = tf.nn.xw_plus_b(drop_out, w_full, b_full)

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
        # w_max = None
        # bp_max = None
        # bm_max = None
        # bn_max = None
        # up_max = None
        # um_max = None
        # un_max = None
        res_max = None
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
                if epoch >= 0:
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
                        loss_test_batch, test_predict, test_label, test_equal, test_acc = sess.run([loss, predict, label, equal, accuracy],
                                                             feed_dict={input_x: x_test,
                                                                        input_y: y_test,
                                                                        input_s: s_test,
                                                                        input_f: f_test,
                                                                        sen_len_ph: test_len,
                                                                        keep_prob_ph: 1.0})
                        accuracy_test += test_acc
                        loss_test += loss_test_batch

                        add_value = []
                        assert len(test_predict) == len(test_label) == len(test_equal)
                        for i in range(len(test_equal)):
                            add_value.append((test_equal[i], test_predict[i], test_label[i]))
                        result_tag.extend(add_value)
                        #out
                        # out_s = s_test[14]
                        # for e in range(len(out_s)):
                        #     emo = ""
                        #     if out_s[e][0] == 1:
                        #         emo = "负的："
                        #     elif out_s[e][1] == 1:
                        #         emo = "中性："
                        #     elif out_s[e][2] == 1:
                        #         emo = "正的："
                        #     print(rev_dic[x_test[14][e]], "：", emo, test_alphas[14][e])
                        # print("-----------------------------------")
                    accuracy_test /= test_batches
                    loss_test /= test_batches
                    if accuracy_test > max_acc:
                        max_acc = accuracy_test
                        res_max = result_tag
                        # print(res_max)
                        # w_max = W.eval()
                        # bp_max = b_pos.eval()
                        # bm_max = b_med.eval()
                        # bn_max = b_neg.eval()
                        # up_max = u_pos.eval()
                        # um_max = u_med.eval()
                        # un_max = u_neg.eval()
                    print("accuracy_test == ", accuracy_test)
                    print("epoch = ", epoch, "max == ", max_acc)


        print("max_accuracy == ", max_acc)
        return max_acc, res_max

max_acc = start_train()