#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from path import *

pos_num = 0
neg_num = 0
med_num = 0
data = dict()
def getSenDict(file_path, low=-0.366, high=0.366, word_index=0, score_index=1):
    global data
    global pos_num, neg_num, med_num
    f = open(file_path, "r")
    for line in f:
        line = line.split('\t')
        word = line[word_index]
        score = float(line[score_index])
        if word in data:
            continue
        if score < low:
            data[word] = 0
            neg_num += 1
        elif score > high:
            data[word] = 2
            pos_num += 1
        else:
            data[word] = 1
            med_num += 1


def getSenDictRev(file_path, low=-0.366, high=0.366, word_index=0, score_index=1):
    global data
    global pos_num, neg_num, med_num
    f = open(file_path, "r")
    for line in f:
        line = line.split('\t')
        word = line[word_index]
        word = word[:len(word) - 1]
        if word[0] == '#':
            word = word[1:]
        score = float(line[score_index])
        if word in data:
            continue
        if score < low:
            data[word] = 0
            neg_num += 1
        elif score > high:
            data[word] = 2
            pos_num += 1
        else:
            data[word] = 1
            med_num += 1

getSenDict("/Users/zrzn/Downloads/SCL-OPP/SCL-OPP.txt", -0.366, 0.366)
getSenDict("/Users/zrzn/Downloads/SCL-NMA/SCL-NMA.txt", -0.366, 0.366)
getSenDictRev("/Users/zrzn/Downloads/SemEval2015-English-Twitter-Lexicon/SemEval2015-English-Twitter-Lexicon.txt", -0.366, 0.366, 1, 0)

print(data)
print(len(data))

print("pos_num == ", pos_num)
print("med_num == ", med_num)
print("neg_num == ", neg_num)
# f_dict = open(all_path + "sen_dict.pkl", "wb")
# pickle.dump(data, f_dict)
