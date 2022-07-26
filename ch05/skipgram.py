# -*- coding: utf-8 -*-
#https://github.com/oreilly-japan/deep-learning-with-keras-ja/blob/master/ch05/keras_cbow.py
from __future__ import print_function

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams

text = "I love green eggs and ham ."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}

wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id), window_size=1)
print(len(pairs), len(labels))
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        id2word[pairs[i][0]], pairs[i][0], 
        id2word[pairs[i][1]], pairs[i][1],
        labels[i]))