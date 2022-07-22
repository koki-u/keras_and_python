import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

def lenet(input_shape, num_classes):
    model = Sequential()
    #フィルタ；20個、特徴マップは28 × 28 × 20 
    model.add(Conv2D(20, kernel_size=5, padding="same",
              input_shape=input_shape, activation="relu"))
    #14 × 14 × 20
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #フィルタ数を増やす。14 × 14 × 50
    model.add(Conv2D(50, kernel_size=5, padding="same", activation="relu"))
    #7 × 7 × 50
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #特徴マップをベクトルに変換、全結合層につなげる準備
    model.add(Flatten())
    #特徴マップをならしたベクトルから、サイズ500のベクトルを出力
    model.add(Dense(500, activation="relu"))
    #クラス数だけベクトルを出力する。
    model.add(Dense(num_classes))
    #softmax関数で、値を0-1に圧縮
    model.add(Activation("softmax"))
    return model

class MNISTDataset():

    def __init__(self):
        self.image_shape = (28, 28, 1)
        self.num_classes = 10

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data

class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        logdir = "logdir_lenet"
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
    
    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        self._target.fit(x_train, y_train, 
                        batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split, 
                        callbacks=[TensorBoard(log_dir=self.log_dir)],
                        verbose=self.verbose)

dataset = MNISTDataset()

#make model
model = lenet(dataset.image_shape, dataset.num_classes)

#train the model
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam())
trainer.train(x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)

#show result
score = model.evaluate(x_test, y_test, verbose=0)
print("test loss:", score[0])
print("test accuracy:", score[1])

