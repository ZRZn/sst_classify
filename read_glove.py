#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from path import *

glove_f = open("/Users/ZRZn1/Downloads/glove.6B/glove.6B.200d.txt", "rb")

emb = []
vocab = dict()
i = 0
for line in glove_f.readlines():
    line = line.decode('utf-8').strip()
    row = line.split(' ')
    vocab[row[0]] = len(vocab)
    emb.append(row[1:])
    if i < 5:
        print(row[0])
        print(row[1:])
    i += 1

emb = np.asarray(emb, dtype="float32")

print(emb[230986])
print(len(emb))
print(len(vocab))

emb_fir = open(all_path + "glove_emb.pkl", "wb")
pickle.dump(emb, emb_fir)
emb_fir.close()

dic_fir = open(all_path + "dic_glove.pkl", "wb")
pickle.dump(vocab, dic_fir)
dic_fir.close()