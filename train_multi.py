#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import main
import pickle
from path import all_path


result = []
max_acc = 0
w_m = None
b_m = None
u_m = None
for i in range(15):
    acc, w_out, b_out, u_out = main.start_train()
    result.append(acc)
    if acc > max_acc:
        max_acc = acc
        w_m = w_out
        b_m = b_out
        u_m = u_out
    print("第 ", i, " 次， acc = ", max_acc)

print("result == ", result)
f = open(all_path + "res.pkl", "wb")
pickle.dump(result, f)
f.close()

f_w = open(all_path + "wbu.pkl", "wb")
pickle.dump(w_m, f_w)
pickle.dump(b_m, f_w)
pickle.dump(u_m, f_w)
f_w.close()