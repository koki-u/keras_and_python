if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

#クラス分類に使用する上位の層は置き換えるので、include_top=False
#これにより、以下のmixe10以降の層は除かれる。
base_model = InceptionV3(weights="imagenet", include_top=False)

#layer.name, layer.input_shpae, layer.output_shape
#None：バッチサイズ
("mixed10", [(None, 8, 8, 320), (None, 8, 8, 768), (None, 8, 8, 768), (None, 8, 8, 192)], (None, 8, 8, 2048))
("avg_pool", (None, 8, 8, 2048), (None, 1, 1, 2048))
("flatten", (None, 1, 1, 2048), (None, 2048))
("predictions", (None, 2048), (None, 1000))

#add a global spatial average pooling layer
x = base_model.output
#GlobalAveragePooling2Dを利用して、(batch_size, rows, cols, channels)の特徴マップを(batch_size, channels)に変換している。
x = GlobalAveragePooling2D()(x)
# add fully-connected layer
x = Dense(1024, activation="relu")(x)
#logistic layer -> 200 classes
predictions = Dense(200, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model ( should be done after setting layers to non-trainable)
model.compile(optimizer="rmsprop", loss="categoraical_crossentropy")

#train the model on the new data for a few epochs
model.fit_generator(...)


#ファインチューニングする
#250層以降をlayer.trainable = True にしてcompileする。
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizers=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(...)