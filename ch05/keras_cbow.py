# -*- coding: utf-8 -*-
#文脈単語の単語IDは、小さいランダムな重みで初期化された共通のEmbedding層に入力される。
#各単語のIDは、Embedding層によってembed_size次元のベクトルに変換される。各行が(2 * window_size, embed_size)サイズの行列に変換される
#そして、すべての分散表現の平均値を計算するLambda層に入力される。
#この平均値は全結合層に入力され、各業に対して、Vocab_size次元の密ベクトルが作成される。
#全結合層の活性化関数はソフトマックス関数で、出力ベクトルの最大値を確率として報告する。
#https://github.com/oreilly-japan/deep-learning-with-keras-ja/blob/master/ch05/keras_cbow.py
from __future__ import division, print_function

from keras.models import Sequential
from keras.layers import Dense, Lambda, Embedding
import keras.backend as K

vocab_size = 5000
embed_size = 300
window_size = 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, 
                    embeddings_initializer='glorot_uniform',
                    input_length=window_size*2))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
model.add(Dense(vocab_size, kernel_initializer='glorot_uniform', 
                activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adadelta")

# get weights
weights = model.layers[0].get_weights()[0]