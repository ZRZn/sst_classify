#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import WordPunctTokenizer
from path import *

word_cut = WordPunctTokenizer()
tokenizer = nltk.data.load(nltk_path)

dic_fir = open(all_path + "dic_glove_sst.pkl", "rb")
dictionary = pickle.load(dic_fir)
dic_fir.close()
print("dic_len == ", len(dictionary))

sen_fir = open(all_path + "default_dic.pkl", "rb")
sen_dic = pickle.load(sen_fir)
sen_fir.close()

score_fir = open(all_path + "score_dic.pkl", "rb")
score_dic = pickle.load(score_fir)
score_fir.close()

pos_num = 0
neg_num = 0
med_num = 0

un_num = 0
yes_num = 0
in_num = 0
out_num = 0

bi = 0
tri = 0

three_count = 0
two_count = 0
# print("pos_num == ", pos_num)
# print("med_num == ", med_num)
# print("neg_num == ", neg_num)
def getSingle(word):
    temp_sen = [0, 0, 0]
    if word not in sen_dic:
        temp_sen[1] = 1
    else:
        temp_sen[sen_dic[word]] = 1
    return temp_sen

def read_data(file_path):
    global un_num, yes_num, three_count, two_count, bi, tri
    data = list()
    sen_data = list()
    score_data = list()
    f = open(file_path, "r")
    for line in f:
        words = word_cut.tokenize(line)
        for i in range(len(words)):
            words[i] = words[i].lower()
        word_int = []
        for word in words:
            if word not in dictionary:
                word_int.append(0)
            else:
                word_int.append(dictionary[word])
        data.append(word_int)

        senti_int = []
        i = 0
        while i < len(words)-2:
            trigram = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
            bigram = words[i] + ' ' + words[i + 1]
            temp_sen = [0, 0, 0]
            if trigram in sen_dic:
                temp_sen[sen_dic[trigram]] = 1
                # senti_int.append(getSingle(words[i]))
                # senti_int.append(getSingle(words[i + 1]))
                senti_int.append(temp_sen)
                senti_int.append(temp_sen)
                senti_int.append(temp_sen)
                i += 3
                yes_num += 3
                three_count += 1
                continue
            elif bigram in sen_dic:
                temp_sen[sen_dic[bigram]] = 1
                # senti_int.append(getSingle(words[i]))
                senti_int.append(temp_sen)
                senti_int.append(temp_sen)
                i += 2
                yes_num += 2
                two_count += 1
                continue
            elif words[i] in sen_dic:
                temp_sen[sen_dic[words[i]]] = 1
                senti_int.append(temp_sen)
                i += 1
                yes_num += 1
                continue
            else:
                un_num += 1
                temp_sen[1] = 1
                senti_int.append(temp_sen)
                i += 1
                continue
        if len(senti_int) == len(words) - 2:
            bigram = words[len(words) - 2] + ' ' + words[len(words) - 1]
            temp_sen = [0, 0, 0]
            if bigram in sen_dic:
                temp_sen[sen_dic[bigram]] = 1
                # senti_int.append(getSingle(words[len(words) - 2]))
                senti_int.append(temp_sen)
                senti_int.append(temp_sen)
                yes_num += 2
                two_count += 1
            else:
                if words[len(words) - 2] in sen_dic:
                    temp_sen[sen_dic[words[len(words) - 2]]] = 1
                    senti_int.append(temp_sen)
                    yes_num += 1
                else:
                    un_num += 1
                    temp_sen[1] = 1
                    senti_int.append(temp_sen)
                temp_sen = [0, 0, 0]
                if words[len(words) - 1] in sen_dic:
                    temp_sen[sen_dic[words[len(words) - 1]]] = 1
                    senti_int.append(temp_sen)
                    yes_num += 1
                else:
                    un_num += 1
                    temp_sen[1] = 1
                    senti_int.append(temp_sen)
        elif len(senti_int) == len(words) - 1:
            temp_sen = [0, 0, 0]
            if words[len(words) - 1] in sen_dic:
                temp_sen[sen_dic[words[len(words) - 1]]] = 1
                senti_int.append(temp_sen)
                yes_num += 1
            else:
                un_num += 1
                temp_sen[1] = 1
                senti_int.append(temp_sen)
        sen_data.append(senti_int)



        i = 0
        score_int = []
        while i < len(words)-2:
            trigram = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
            bigram = words[i] + ' ' + words[i + 1]
            if trigram in score_dic:
                score_int.append(score_dic[words[i]])
                score_int.append(score_dic[words[i + 1]])
                score_int.append(score_dic[trigram])
                i += 3
                tri += 1
                continue
            elif bigram in score_dic:
                score_int.append(score_dic[words[i]])
                score_int.append(score_dic[bigram])
                i += 2
                bi += 1
                continue
            elif words[i] in score_dic:
                score_int.append(score_dic[words[i]])
                i += 1
                continue
            else:
                score_int.append(0.5)
                i += 1
                continue
        if len(score_int) == len(words) - 2:
            bigram = words[len(words) - 2] + ' ' + words[len(words) - 1]
            if bigram in score_dic:
                score_int.append(score_dic[words[len(words) - 2]])
                score_int.append(score_dic[bigram])
                bi += 1
            else:
                if words[len(words) - 2] in score_dic:
                    score_int.append(score_dic[words[len(words) - 2]])
                else:
                    score_int.append(0.5)
                if words[len(words) - 1] in score_dic:
                    score_int.append(score_dic[words[len(words) - 1]])
                else:
                    score_int.append(0.5)
        elif len(score_int) == len(words) - 1:
            if words[len(words) - 1] in score_dic:
                score_int.append(score_dic[words[len(words) - 1]])
            else:
                score_int.append(0.5)
        score_data.append(score_int)

        print(words, ' | ', score_int)
    f.close()

    assert len(data) == len(sen_data) == len(score_data)
    for i in range(len(data)):
        if len(data[i]) != len(sen_data[i]) or len(score_data[i]) != len(sen_data[i]):
            print("出错啦！！！datasize = %d, sen_data = %d", len(data[i]), len(sen_data[i]))
    return data, sen_data, score_data


def read_y(file_path):
    y = []
    f = open(file_path, 'r')
    for line in f:
        temp_y = [0, 0, 0, 0, 0]
        index = int(line)
        assert index < 5
        temp_y[index] = 1
        y.append(temp_y)
    return y


train_x, train_s, train_f = read_data(origin_path + "train.txt")
test_x, test_s, test_f = read_data(origin_path + "test.txt")

# print(test_f)
train_y = read_y(origin_path + "train_label.txt")
test_y = read_y(origin_path + "test_label.txt")

print("yes_num == ", yes_num)
print("un_num == ", un_num)
pos = 0
neg = 0
med = 0
for s in train_s:
    for score in s:
        if score[0] == 1:
            neg += 1
        elif score[1] == 1:
            med += 1
        else:
            pos += 1
print("pos == ", pos)
print("neg == ", neg)
print("med == ", med)
print("three_count == ", three_count)
print("two_count == ", two_count)
print("bi == ", bi)
print("tri == ", tri)

train_fir = open(all_path + "train.pkl", "wb")
test_fir = open(all_path + "test.pkl", "wb")

pickle.dump(train_x, train_fir)
pickle.dump(train_y, train_fir)
pickle.dump(train_s, train_fir)
pickle.dump(train_f, train_fir)

pickle.dump(test_x, test_fir)
pickle.dump(test_y, test_fir)
pickle.dump(test_s, test_fir)
pickle.dump(test_f, test_fir)

train_fir.close()
test_fir.close()