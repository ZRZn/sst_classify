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
    if len(line[0].split(' ')) <= 2:
        sen_dict[line[0].lower()] = int(line[1])

print(sen_dict)

dic_f = open(all_path + "default_dic.pkl", 'wb')
pickle.dump(sen_dict, dic_f)
dic_f.close()


# f_l = open(origin_path + "sentiment_labels.txt", 'r')
# f_d = open(origin_path + "dictionary.txt", 'r')
#
# tmp_dict = dict()
#
# for line in f_l:
#     line = line.strip().split('|')
#     tmp_dict[line[0]] = float(line[1])
#
# # print(tmp_dict)
# score_dict = dict()
# for line in f_d:
#     line = line.strip().split('|')
#     if len(line[0].split()) <= 3:
#         score_dict[line[0]] = tmp_dict[line[1]]
# print(len(score_dict))
#
# max_sen = 0
# min_sen = 0
#
# # print(score_dict.values())
# for score in score_dict.values():
#     if score > max_sen:
#         max_sen = score
#     if score < min_sen:
#         min_sen = score
#
# print(max_sen)
# print(min_sen)
#
# f_save = open(all_path + "score_dic.pkl", 'wb')
# pickle.dump(score_dict, f_save)
# f_save.close()