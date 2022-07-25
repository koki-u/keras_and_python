from keras.layers import Merge, Dense, Reshape, Embedding
from keras.models import Sequential

# how many words do u use ?
vocab_size = 5000
#dim 
embed_size = 300

#中心語の用のモデル
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,
               embeddings_initializer="glorot_uniform",
               input_length=1))
word_model.add(Reshape((embed_size, )))

#文脈表現用のモデル
context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,
                  embeddings_initializer="glorot_uniform",
                  input_length=1))
context_model.add(Reshape((embed_size,)))

#この二つのモデルに対して、内積を計算する。
#計算結果を全結合層に入力する。

#出力を計算する
#平均二乗誤差を用いているのは、内積計算をしたときに、類似しているベクトルは大きくなる。
#シグモイドを通した結果は、1に近づく。
model = Sequential()
model.add(Merge([word_model, context_model], mode="dot"))
model.add(Dense(1, kernel_initializer="glorot_uniform",
          activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam")

