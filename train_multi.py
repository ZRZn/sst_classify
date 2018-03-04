#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main import start_train
import pickle
from path import all_path


f = open(all_path + "result.pkl", "wb")

result = []
for i in range(10):
    acc = start_train()
    result.append(acc)

pickle.dump(result, f)
f.close()