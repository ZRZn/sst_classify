#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from path import *
import numpy as np

emd_file = open(all_path + "glove_emb.pkl", "rb")
emb_array = pickle.load(emd_file)
emd_file.close()

print(type(emb_array))

dic_fir = open(all_path + "dic2.pkl", "rb")
dic_sst = pickle.load(dic_fir)
dic_fir.close()

print("删除前 = ", len(dic_sst))

dic_fir2 = open(all_path + "dic_glove.pkl", "rb")
dic_imdb = pickle.load(dic_fir2)
dic_fir2.close()
print("imdb len == ", len(emb_array))

words_sst = dic_sst.keys()
words_imdb = dic_imdb.keys()
print(words_sst)
no_words = []
for word in words_sst:
    if word not in words_imdb:
        no_words.append(word)

for no_word in no_words:
    dic_sst.pop(no_word)
print("删除后 = ", len(dic_sst))
dic = sorted(dic_sst.items(), key=lambda item: item[1])
print(dic)

emb_res = []
i = 0
dic_copy = []
for it in dic:
    dic_copy.append((it[0], i))
    i += 1
print(dic_copy)


for it in dic_copy:
    print(dic_imdb[it[0]])
    emb_res.append(emb_array[dic_imdb[it[0]]])

print(len(emb_res))

print(emb_res[10892])

emb_res = np.array(emb_res)

print("emb_res type == ", type(emb_res))

dic_fir3 = open(all_path + "emb_array22.pkl", "wb")
pickle.dump(emb_res, dic_fir3)
dic_fir3.close()

dic_copy = dict(dic_copy)
dic_fir4 = open(all_path + "dic_glove_sst.pkl", "wb")
pickle.dump(dic_copy, dic_fir4)
dic_fir4.close()