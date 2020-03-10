from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
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
parser.add_argument('--batch_size', type=int, default=164, help='number of instances per batch')
parser.add_argument('--kernel_size', type=int, default=(3, 3), help='kernel size for model')
parser.add_argument('--drop', type=float, default=0.35, help='dropout prob')
parser.add_argument('--name', type=str, default='sub_model.h5', help='specify name of saved model' )

FLAGS = parser.parse_args()

EPOCHS = FLAGS.epochs
VAL_SIZE = FLAGS.val_size
BATCH_SIZE = FLAGS.batch_size
KERNEL_SIZE = FLAGS.kernel_size
DROP = FLAGS.drop
NAME = FLAGS.name


images_training = ImageDataGenerator().flow_from_directory('Images', target_size=(225, 150),
                                                  classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer',
                                                            'Roteiche'],
                                                  batch_size=5550,
                                                  color_mode="grayscale",
                                                  shuffle=True)

images_test = ImageDataGenerator().flow_from_directory('Test_Images', target_size=(225, 150),
                                                  classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer',
                                                            'Roteiche'],
                                                  batch_size=1350,
                                                  color_mode="grayscale",
                                                  shuffle=True)

x_train, y_train = next(images_training)
x_test, y_test = next(images_test)
num_classes = 7

x_test = x_test/255
x_train = x_train/255

labs = y_train.argmax(axis=1)

idx_eiche = np.argwhere(labs == 2)
idx_esche = np.argwhere(labs == 3)
idx_kiefer = np.argwhere(labs == 5)

idx = np.concatenate([idx_eiche, idx_esche, idx_kiefer])

idx = idx[:,-1]

x_small = x_train[idx, :, :, :]
y_small = y_train[idx,:]

rotate = ImageDataGenerator(rotation_range=15)
vert = ImageDataGenerator(width_shift_range=[-15, 15])
hor = ImageDataGenerator(height_shift_range=0.2)
rotate.fit(x_small)
vert.fit(x_small)
hor.fit(x_small)

for x_rot_s, y_rot_s in rotate.flow(x_small, y_small, batch_size=700, shuffle=True):
    break

for x_vert_s, y_vert_s in vert.flow(x_small, y_small, batch_size=700, shuffle=True):
    break

for x_hor_s, y_hor_s in hor.flow(x_small, y_small, batch_size=700, shuffle=True):
    break

x_small = np.concatenate((x_small, x_rot_s), axis=0)
y_small = np.concatenate((y_small, y_rot_s), axis=0)

x_small = np.concatenate((x_small, x_vert_s), axis=0)
y_small = np.concatenate((y_small, y_vert_s), axis=0)

x_small = np.concatenate((x_small, x_hor_s), axis=0)
y_small = np.concatenate((y_small, y_hor_s), axis=0)

x_train = np.concatenate((x_train, x_small), axis=0)
y_train = np.concatenate((y_train, y_small), axis=0)




rotate.fit(x_train)

for x_rot, y_rot in rotate.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

vert.fit(x_train)

for x_vert, y_vert in vert.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

hor.fit(x_train)

for x_hor, y_hor in hor.flow(x_train, y_train, batch_size=2000, shuffle=True):
    break

x_train = np.concatenate((x_train, x_rot), axis=0)
y_train = np.concatenate((y_train, y_rot), axis=0)

x_train = np.concatenate((x_train, x_vert), axis=0)
y_train = np.concatenate((y_train, y_vert), axis=0)

x_train = np.concatenate((x_train, x_hor), axis=0)
y_train = np.concatenate((y_train, y_hor), axis=0)

x_train = x_train[:, 40:175, 20:120, :]
x_test = x_test[:, 40:175, 20:120, :]

#sometimes = lambda aug: iaa.Sometimes(1.0, aug)
#
## Define our sequence of augmentation steps that will be applied to every image.
#seq = iaa.Sequential(
#    [
#
#        # crop some of the images by 0-10% of their height/width
#        sometimes(iaa.Crop(percent=(0, 0.05))),
#
#        # Apply affine transformations to some of the images
#        # - scale to 80-120% of image height/width (each axis independently)
#        # - translate by -20 to +20 relative to height/width (per axis)
#        # - rotate by -45 to +45 degrees
#        # - shear by -16 to +16 degrees
#        # - order: use nearest neighbour or bilinear interpolation (fast)
#        # - mode: use any available mode to fill newly created pixels
#        #         see API or scikit-image for which modes are available
#        # - cval: if the mode is constant, then use a random brightness
#        #         for the newly created pixels (e.g. sometimes black,
#        #         sometimes white)
#        sometimes(iaa.Affine(
#            scale={"x": (0.995, 1.005), "y": (0.995, 1.005)},
#            translate_percent={"x": (-0.005, 0.005), "y": (-0.005, 0.005)},
#            order=[0, 1],
#            #cval=(0, 255),
#            mode=ia.ALL
#        )),
#
#        #
#        # Execute 0 to 5 of the following (less important) augmenters per
#        # image. Don't execute all of them, as that would often be way too
#        # strong.
#        #
#        iaa.SomeOf((0, 2),
#            [
#
#
#                # Blur each image with varying strength using
#                # gaussian blur (sigma between 0 and 3.0),
#                # average/uniform blur (kernel size between 2x2 and 7x7)
#                # median blur (kernel size between 3x3 and 11x11).
#                iaa.OneOf([
#                    iaa.GaussianBlur((0, 0.005)),
#                    iaa.AverageBlur(k=(0.1, 0.15)),
#                ]),
#
#                # Sharpen each image, overlay the result with the original
#                # image using an alpha between 0 (no sharpening) and 1
#                # (full sharpening effect).
#                iaa.Sharpen(alpha=(0, 0.1), lightness=(0.95, 1)),
#
#                # Same as sharpen, but for an embossing effect.
#                iaa.Emboss(alpha=(0, 0.1), strength=(0, 0.1)),
#
#                # Improve or worsen the contrast of images.
#                iaa.LinearContrast((0.97, 1.02), per_channel=0.01),
#
#
#                # In some images distort local areas with varying strength.
#                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02)))
#            ],
#            # do all of the above augmentations in random order
#            random_order=False
#        )
#    ],
#    # do all of the above augmentations in random order
#    random_order=False
#)
#
#images_aug = seq(images=x_train)
#
#x_train = np.concatenate((x_train, images_aug), axis=0)
#y_train = np.concatenate((y_train, y_train), axis=0)
#
#print(x_train.shape)
#print(y_train.shape)
#plt.imshow(x_train[20015,:,:,0])
#plt.show()
#plt.imshow(x_train[20014,:,:,0])
#plt.show()
#plt.imshow(x_train[20005,:,:,0])
#plt.show()
#plt.imshow(x_train[23015,:,:,0])
#plt.show()
#plt.imshow(x_train[21015,:,:,0])
#plt.show()
#plt.imshow(x_train[25015,:,:,0])
#plt.show()
#plt.imshow(x_train[24015,:,:,0])
#plt.show()
#plt.imshow(x_train[24515,:,:,0])
#plt.show()
#plt.imshow(x_train[25515,:,:,0])
#plt.show()
#plt.imshow(x_train[21215,:,:,0])
#plt.show()

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

model.save(NAME)



