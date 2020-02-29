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



images_training = ImageDataGenerator().flow_from_directory('Images', target_size=(150, 225),
                                                  classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer',
                                                            'Roteiche'],
                                                  batch_size=5550,
                                                  color_mode="grayscale",
                                                  shuffle=True)

images_test = ImageDataGenerator().flow_from_directory('Test_Images', target_size=(150, 225),
                                                  classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer',
                                                            'Roteiche'],
                                                  batch_size=1350,
                                                  color_mode="grayscale",
                                                  shuffle=True)

x_train, y_train = next(images_training)
x_test, y_test = next(images_test)

x_test = x_test/255
x_train = x_train/255

#x_train, x_random, y_train, y_random = train_test_split(imgs, labels, test_size=0, random_state=101)
#x_test, x_random1, y_test,  y_random1 = train_test_split(test_imgs, test_labels, test_size=0, random_state=101)
#print(x_random.shape)
print(x_train.shape)
print(x_test.shape)

rotate = ImageDataGenerator(rotation_range=15)
rotate.fit(x_train)

for x_rot, y_rot in rotate.flow(x_train, y_train, batch_size=1500, shuffle=True):
    break

x_train = np.concatenate((x_train, x_rot), axis=0)
y_train = np.concatenate((y_train, y_rot), axis=0)

batch_size = 30
num_classes = 7
epochs = 20
es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

print(x_train.shape)

model = Sequential()
model.add(Conv2D(4, kernel_size=(20,20), activation='relu', input_shape=(150, 225, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size=(32,32), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size = (12, 12), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#test model on validation data - split = 0.2
history = model.fit(x_train, y_train, validation_split=0.2,
                   epochs=epochs, batch_size=batch_size,
                    callbacks=[es])

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(acc * 100)

model.save('2D_subspecies_model_new.h5')

