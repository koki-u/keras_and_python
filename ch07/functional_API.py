# Kerasのfunctional APIはネットワークを構成する各層を関数として定義し、複雑なネットワークを構築できる。
# https://keras.io/getting-started/sequential-model-guide/ より抜粋されたもの。 

from keras.models import Sequential
from keras.layers import Dense, Activation

# xは、(None, 784)で任意の長さで、バッチサイズに相当する。
# yは、(None, 10)
model = Sequential([Dense(32, input_shape=(784,)), 
                    Activation('relu'), 
                    Dense(10), 
                    Activation('softmax')])


# functional APIを用いて、書き直すと以下のようになる。
from keras.layers import Input, Dense, Activation
from keras.models import Model

#入力
x = Input(shape=(784,))

#関数を定義
g = Dense(32)
s_2 = Activation("sigmoid")
f = Dense(10)
s_K = Activation("softmax")
y = s_K(f(s_2(g(x))))

model = Model(inputs=x, outputs=y)
model.compile(loss="categorical_crossentropy", optimizer="adam")

#このように定義されたモデルは、関数の組み合わせとしてみなすことができる。
#　学習済みのモデルも関数として見ることができる。
#　学習済みの画像分類モデルを時系列の入力を扱うTimeDistributedにくみこんで、複数の画像処理をする例がある。

