if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.make_tensorboard import make_tensorboard

import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1671)

#network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT=0.2
DROPOUT = 0.3

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#28 pix × 28 pix = 784 
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

#10 outputs
#final stage is softmax
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

#MSE/バイナリクロスエントロピー/カテゴリカルクロスエントロピーが損失関数としてよく使われている。
model.compile(loss='categorical_crossentropy', 
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V3')]
        
model.fit(X_train, Y_train, 
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=callbacks, 
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
#損失関数によって得られる最小値
print("\nTest score:", score[0])
#評価関数によって得られる最良値
print("Test accuracy:", score[1])

#--------------------------------------------------------------

#学習データの精度は評価データの精度を超えるべき
#そうでない場合は、充分に学修していないことになる。→学習回数を増やす！

#学習データと評価データの精度がエポック数に対してどのように増加するかを観察するのが大事

#隠れ層にドロップアウトを使用すると、汎化性能が向上する。