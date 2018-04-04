#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from path import *

f = open(origin_path + "senti_lexicon.txt", 'r')
i = 0
sen_dict = dict()
num_dict = [0, 0, 0]
for line in f:
    line = line.strip()
    line = line.split('==')
    # print(line[0])
    if line[1][0] == '=':
        # print("before == ", line[1])
        line[1] = line[1][1:]
        # print("after == ", line[1])
    if len(line[0].split(' ')) < 4:
        sen_dict[line[0].lower()] = int(line[1])
        num_dict[int(line[1])] += 1
        # print(line[0].lower(), " | ", int(line[1]))

print(len(sen_dict))
print("num_dict == ", num_dict)
# print(sen_dict)

dic_f = open(all_path + "default_dic.pkl", 'wb')
pickle.dump(sen_dict, dic_f)
dic_f.close()

# sen_dict = dict()

f_l = open(origin_path + "sentiment_labels.txt", 'r')
f_d = open(origin_path + "dictionary.txt", 'r')

tmp_dict = dict()

for line in f_l:
    line = line.strip().split('|')
    tmp_dict[line[0]] = float(line[1])

# print(tmp_dict)
num_dict = [0, 0, 0]
score_dict = dict()
for line in f_d:
    line = line.strip().split('|')
    if len(line[0].split()) == 1:
        score_dict[line[0]] = tmp_dict[line[1]]
        if tmp_dict[line[1]] < 0.5:
            num_dict[0] += 1
            # sen_dict[line[0]] = 0
        elif tmp_dict[line[1]] == 0.5:
            num_dict[1] += 1
            # sen_dict[line[0]] = 1
        else:
            num_dict[2] += 1
            # sen_dict[line[0]] = 2
print(len(score_dict))

print("num_dict == ", num_dict)

print("测试 ", sen_dict['not'])

# print(score_dict.values())

#
# f_save = open(all_path + "score_dic.pkl", 'wb')
# pickle.dump(score_dict, f_save)
# f_save.close()

# dic_f = open(all_path + "default_dic.pkl", 'wb')
# pickle.dump(sen_dict, dic_f)
# dic_f.close()
