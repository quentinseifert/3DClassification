from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#example prediction
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
image_list = []

for filename in glob.glob('C:/Users/anton/Desktop/bad_trees/Hainbuche/*.png'):
    test_image = image.load_img(filename, target_size=(320, 240), color_mode="grayscale")
    test_image = image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    image_list.append(test_image)

images = np.array(image_list)

results = []

for i in range(0, images.shape[0]):
    temp = model.predict_proba(images[i,:])
    results.append(temp)

print(results)


# evaluate model

#images = ImageDataGenerator().flow_from_directory('Laub_Nadel', target_size=(320, 240),
#                                                  classes=['laub', 'nadel'],
#                                                  batch_size=4420,
#                                                  color_mode="grayscale",
#                                                  shuffle=True)
#
#imgs, labels = next(images)
#imgs = imgs/255
#
#x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.3, random_state=101)
#
#model = load_model('2Dmodel.h5')
#
#loss, acc = model.evaluate(x_test, y_test, verbose=0)
#print(acc * 100)


