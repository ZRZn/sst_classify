#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main import start_train
import pickle
from path import all_path


result = []
for i in range(10):
    acc = start_train()
    result.append(acc)

print("result == ", result)
f = open(all_path + "multi_res.pkl", "wb")
pickle.dump(result, f)
f.close()