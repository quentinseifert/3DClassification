from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--val_size', type=float, default=0.2, help='size of validation data')
parser.add_argument('--batch_size', type=int, default=100, help='number of instances per batch')
parser.add_argument('--kernel_size', type=int, default=(5, 5), help='kernel size for model')
parser.add_argument('--drop', type=float, default=0.5, help='dropout prob')

FLAGS = parser.parse_args()

EPOCHS = FLAGS.epochs
VAL_SIZE = FLAGS.val_size
BATCH_SIZE = FLAGS.batch_size
KERNEL_SIZE = FLAGS.kernel_size
DROP = FLAGS.drop


images_training = ImageDataGenerator().flow_from_directory('Laub_vs_Nadel', target_size=(225, 150),
                                                  classes=['Laub', 'Nadel'],
                                                  batch_size=5780,
                                                  color_mode="grayscale",
                                                  shuffle=True)

images_test = ImageDataGenerator().flow_from_directory('Test_Laub_vs_Nadel', target_size=(225, 150),
                                                  classes=['Laub', 'Nadel'],
                                                  batch_size=1350,
                                                  color_mode="grayscale",
                                                  shuffle=True)

x_train, y_train = next(images_training)
x_test, y_test = next(images_test)
num_classes = 2

x_test = x_test/255
x_train = x_train/255

rotate = ImageDataGenerator(rotation_range=15)
rotate.fit(x_train)

for x_rot, y_rot in rotate.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

vert = ImageDataGenerator(width_shift_range=[-15, 15])
vert.fit(x_train)

for x_vert, y_vert in vert.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

hor = ImageDataGenerator(height_shift_range=0.2)
hor.fit(x_train)

for x_hor, y_hor in hor.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

x_train = np.concatenate((x_train, x_rot), axis=0)
y_train = np.concatenate((y_train, y_rot), axis=0)

x_train = np.concatenate((x_train, x_vert), axis=0)
y_train = np.concatenate((y_train, y_vert), axis=0)

x_train = np.concatenate((x_train, x_hor), axis=0)
y_train = np.concatenate((y_train, y_hor), axis=0)

x_train = x_train[:, 50:170, 20:120, :]
x_test = x_test[:, 50:170, 20:120, :]

print(x_train.shape)
plt.imshow(x_train[1,:,:,0])
plt.show()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SIZE, shuffle=True)

es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model = Sequential()
model.add(Conv2D(8, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=KERNEL_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=KERNEL_SIZE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#test model on validation data - split = 0.2
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[es])

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(acc * 100)

model.save('2D_Basic_augm.h5')

