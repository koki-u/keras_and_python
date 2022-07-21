import keras
from keras import Sequential

#ここでは、以下の2点を使用する。

#Sequential モデル
#ch01のSequentialの部分を参照

#functional API
#有効非循環グラフ、共有レイヤーモデル、複数出力モデルなどの複雑なモデルを定義できる。

#====================================================
#事前定義ずみのNN層
#====================================================

#通常の全結合層
#keras.layers.Dense()

#RNN(LSTM, GRU)
#keras.layers.RNN()
#keras.layers.SimpleRNN()
#keras.layers.GRU()
#keras.layers.LSTM()

#畳み込み層とプーリング層
#keras.layers.Conv1D()
#keras.layers.Conv2D()
#keras.layers.MaxPooling1D()
#keras.layers.MaxPooling2D()

#正則化層
#kernel_regularizer(重み行列に適用する正則化関数)
#bias_regularizer(バイアスベクトルに適用される正則化関数)
#activity_regularizer(層の出力に適用される正則化関数)

#Dropout
#keras.layers.Dropout(rate, noise_shape=None, seed=None)

#バッチ正規化(学習を加速し、一般的に性能を向上させることのできる手法)
#keras.layers.BatchNormalization()

#========================================================
#モデルの保存と読み込み
#========================================================

#----------------------------------------------------
#モデルアーキテクチャ
#----------------------------------------------------
#JSON形式で保存
#json_string = model.to_json()
#YMAL形式で保存
#yaml_string = model.to_yaml()

#from keras.models import model_from_json
# JSONからのモデル再構築
#model = model_from_json(json_string)
# YMALからのモデル再構築
#model = model_from_yaml(yaml_string)

#----------------------------------------------------
#モデルパラメータ
#-----------------------------------------------------

#from keras.models import load_model
# HDF5形式のファイル'my_model.h5' を作成
#model.save('my_model.h5')
# モデルの削除
#del model
# モデルの読み込み
# model = load_model('my_model.h5')

#=========================================================
#kerasのcallback機能
#=========================================================

#評価関数の値が改善されなくなったら学習を停止して、その履歴を残しておける
keras.callbacks.EarlyStopping()

class LossHistroy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#model = Sequential()
#model.add(Dens....
#....
#history = LossHistroy()
#model.fit(...., callbacks=[history])

#のように使う

