import numpy as np
from keras.preprocessing.image import load_img

def prep_prediction(path):
    img = load_img(path, color_mode='grayscale', target_size=(320, 240))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img

