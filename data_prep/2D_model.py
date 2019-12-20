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


images = ImageDataGenerator().flow_from_directory('Images',
                                                  target_size=(320, 240), classes=['laub', 'nadel'], batch_size=1600)


imgs, labels = next(images)
print(imgs.shape)
print(labels.shape)

#plt.imshow(imgs[1], cmap=plt.cm.binary)
#plt.show()
imgs = imgs/255

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.3, random_state = 101)

batch_size = 30
num_classes = 2
epochs = 50

print(x_train.shape)
print(x_train[3])


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(320, 240, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# test model on validation data - split = 0.3
history = model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=batch_size)

