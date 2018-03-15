#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from path import *

f = open(origin_path + "senti_lexicon.txt", 'r')
i = 0
sen_dict = dict()
for line in f:
    line = line.strip()
    line = line.split('==')
    # print(line[0])
    if line[1][0] == '=':
        print("before == ", line[1])
        line[1] = line[1][1:]
        print("after == ", line[1])
    sen_dict[line[0].lower()] = int(line[1])

print(sen_dict)

dic_f = open(all_path + "default_dic.pkl", 'wb')
pickle.dump(sen_dict, dic_f)
dic_f.close()