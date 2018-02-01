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

dic_fir = open(all_path + "dic.pkl", "rb")
dictionary = pickle.load(dic_fir)
dic_fir.close()

def read_data(file_path):
    data = list()
    f = open(file_path, "r")
    for line in f:
        data.append(word_cut.tokenize(line))
    f.close()
    return data


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

train_x = read_data(origin_path + "train.txt")
train_y = read_y(origin_path + "train_label.txt")

for i in range(1000, 1100):
    print(train_y[i])