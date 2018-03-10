#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
from path import *

pos_num = 0
neg_num = 0
med_num = 0

def getSenDict(data, file_path, low=-0.366, high=0.366, word_index=0, score_index=1):
    global pos_num, neg_num, med_num
    f = open(file_path, "r")
    for line in f:
        line = line.split('\t')
        word = line[word_index]
        score = float(line[score_index])
        if word in data:
            continue
        if score < low:
            data[word] = 0
            neg_num += 1
        elif score > high:
            data[word] = 2
            pos_num += 1
        else:
            data[word] = 1
            med_num += 1
    return data


def getSenDictRev(data, file_path, low=-0.366, high=0.366, word_index=0, score_index=1):
    global pos_num, neg_num, med_num
    f = open(file_path, "r")
    for line in f:
        line = line.split('\t')
        word = line[word_index]
        word = word[:len(word) - 1]
        if word[0] == '#':
            word = word[1:]
        score = float(line[score_index])
        if word in data:
            continue
        if score < low:
            data[word] = 0
            neg_num += 1
        elif score > high:
            data[word] = 2
            pos_num += 1
        else:
            data[word] = 1
            med_num += 1
    return data

sen_dic = dict()

sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-OPP/SCL-OPP.txt", -0.6, 0.40)
sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-NMA/SCL-NMA.txt", -0.52, 0.52)
sen_dic = getSenDictRev(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SemEval2015-English-Twitter-Lexicon/SemEval2015-English-Twitter-Lexicon.txt", -0.52, 0.52, 1, 0)

sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.78, 0.80199)
sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.55, 1.1)
sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.7, 0.71)
sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.27, 1.0)



# print("pos_num == ", pos_num)
# print("med_num == ", med_num)
# print("neg_num == ", neg_num)
#
# print(", == ", sen_dic['...'])
f_dict = open(all_path + "sen_dict.pkl", "wb")
pickle.dump(sen_dic, f_dict)


#pos + neg = med
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-OPP/SCL-OPP.txt", -0.47, 0.30)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-NMA/SCL-NMA.txt", -0.4, 0.4)
# sen_dic = getSenDictRev(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SemEval2015-English-Twitter-Lexicon/SemEval2015-English-Twitter-Lexicon.txt", -0.38, 0.38, 1, 0)
#
# sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.451, 0.64)
# sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.23, 0.865)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.39, 0.46)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.02, 0.844)


# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-OPP/SCL-OPP.txt", -0.6, 0.40)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SCL-NMA/SCL-NMA.txt", -0.52, 0.52)
# sen_dic = getSenDictRev(sen_dic, "/Users/ZRZn1/Downloads/sentiment_lexicon_set/SemEval2015-English-Twitter-Lexicon/SemEval2015-English-Twitter-Lexicon.txt", -0.52, 0.52, 1, 0)
#
# sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.78, 0.80199)
# sen_dic = getSenDict(sen_dic,"/Users/ZRZn1/Desktop/lexicon/Yelp-restaurant-reviews/Yelp-restaurant-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.55, 1.1)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt", -0.7, 0.71)
# sen_dic = getSenDict(sen_dic, "/Users/ZRZn1/Desktop/lexicon/Amazon-laptop-electronics-reviews/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-bigrams.txt", -0.27, 1.0)
# print(sen_dic)
