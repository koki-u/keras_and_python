if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras.preprocessing.image as Image
import keras
import numpy as np

model = VGG16(weights="imagenet", include_top=True)

image_path = "Keras_and_Python/ch03/steaming_train.png"
#本では、Image.load_img() となっていたが、存在しないようだ。
#keras.utils.load_img()　により解決した。
image = keras.utils.load_img(image_path, target_size=(224, 224))

#Image.img_to_array は同様に無理だったので、keras.utils.img_to_array()
x = keras.utils.img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

result = model.predict(x)
result = decode_predictions(result, top=3)[0]
print(result[0][1])
