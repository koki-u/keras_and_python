if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import keras
import numpy as np

base_model = VGG19(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("block4_pool").output)

img_path = "Keras_and_Python/ch03/elephant.jpeg"
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
