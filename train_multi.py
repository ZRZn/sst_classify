#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import main
import pickle
from path import all_path


# result = []
# max_acc = 0
# w_m = None
# b_p = None
# b_m = None
# b_n = None
# u_p = None
# u_m = None
# u_n = None
# for i in range(15):
#     acc, w_max, bp_max, bm_max, bn_max, up_max, um_max, un_max = main.start_train()
#     result.append(acc)
#     if acc > max_acc:
#         max_acc = acc
#         w_m = w_max
#         b_p = bp_max
#         b_m = bm_max
#         b_n = bn_max
#         u_p = up_max
#         u_m = um_max
#         u_n = un_max
#     print("第 ", i, " 次， acc = ", max_acc)
#
# print("result == ", result)
# f = open(all_path + "res.pkl", "wb")
# pickle.dump(result, f)
# f.close()
#
# f_w = open(all_path + "wbu.pkl", "wb")
# pickle.dump(w_m, f_w)
# pickle.dump(b_p, f_w)
# pickle.dump(b_m, f_w)
# pickle.dump(b_n, f_w)
# pickle.dump(u_p, f_w)
# pickle.dump(u_m, f_w)
# pickle.dump(u_n, f_w)
# f_w.close()



result = []
max_acc = 0
max_tag = None
for i in range(15):
    acc, res_tag = main.start_train()
    result.append(acc)
    if acc > max_acc:
        max_acc = acc
        max_tag = res_tag
    print("第 ", i, " 次， acc = ", acc)
print("result == ", result)

f = open(all_path + "res_att.pkl", "wb")
pickle.dump(result, f)
f.close()
