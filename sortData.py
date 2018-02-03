#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from path import *
import pickle

train_fir = open(all_path + "train.pkl", "rb")
test_fir = open(all_path + "test.pkl", "rb")

train_x = pickle.load(train_fir)
train_y = pickle.load(train_fir)
train_s = pickle.load(train_fir)

test_x = pickle.load(test_fir)
test_y = pickle.load(test_fir)
test_s = pickle.load(test_fir)

train_fir.close()
test_fir.close()

assert len(train_x) == len(train_y) == len(train_s)

assert len(test_x) == len(test_y) == len(test_s)



# for i in range(len(train_x)):
#     print(train_x[i])

def sortData(x, y, s, BATCH_SIZE=64):
    for i in range(len(x)):
        if len(x[i]) != len(s[i]):
            print("出错了！！！")
        x[i].append(i)

    x = sorted(x, key=lambda t: len(t))
    res_y = [0] * len(y)
    res_s = [0] * len(s)
    for i in range(len(x)):
        index = x[i].pop()
        res_y[i] = y[index]
        res_s[i] = s[index]
    # for i in range(len(res_y)):
    #     print(res_y[i])
    temp_sen = [0, 1, 0]
    num = len(x) // BATCH_SIZE
    for i in range(num):
        max = len(x[(i + 1) * BATCH_SIZE - 1])
        for j in range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE):
            if max == len(x[j]):
                continue
            for t in range(max - len(x[j])):
                x[j].append(0)
                res_s[j].append(temp_sen)
    max_last = len(x[len(x) - 1])
    last = len(x) % BATCH_SIZE
    for i in range(len(x) - last, len(x)):
        for t in range(max_last - len(x[i])):
            x[i].append(0)
            res_s[i].append(temp_sen)
    return x, res_y, res_s

train_x, train_y, train_s = sortData(train_x, train_y, train_s)

test_x, test_y, test_s = sortData(test_x, test_y, test_s)


# train_fir = open(all_path + "train_out.pkl", "wb")
# test_fir = open(all_path + "test_out.pkl", "wb")
#
# pickle.dump(train_x, train_fir)
# pickle.dump(train_y, train_fir)
# pickle.dump(train_s, train_fir)
#
# pickle.dump(test_x, test_fir)
# pickle.dump(test_y, test_fir)
# pickle.dump(test_s, test_fir)
#
# train_fir.close()
# test_fir.close()



