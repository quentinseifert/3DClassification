from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

##example prediction
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#image_list = []
#
#for filename in glob.glob('C:/Users/anton/Desktop/bad_trees/Hainbuche/*.png'):
#    test_image = image.load_img(filename, target_size=(320, 240), color_mode="grayscale")
#    test_image = image.img_to_array(test_image) / 255
#    test_image = np.expand_dims(test_image, axis=0)
#    image_list.append(test_image)
#
#images = np.array(image_list)
#
#results = []
#
#for i in range(0, images.shape[0]):
#    temp = model.predict_proba(images[i,:])
#    results.append(temp)
#
#print(results)
#
#
#evaluate model

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



model = load_model('2D_subspecies_model_augment1.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

images_test = ImageDataGenerator().flow_from_directory('Test_Images', target_size=(225, 150),
                                                  classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer',
                                                            'Roteiche'],
                                                  batch_size=1350,
                                                  color_mode="grayscale",
                                                  shuffle=True)

x_test, y_test = next(images_test)



x_test = x_test/255

x_test = x_test[:, 40:175, 20:120, :]
loss, acc = model.evaluate(x_test, y_test)
print(acc)

y_pred = model.predict_classes(x_test)

#y_pred = label_binarizer.fit_transform(y_pred)

print(y_pred.shape)
print(y_test.shape)
print(y_test)
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

y_test = first_nonzero(y_test, axis=1, invalid_val=-1)

print(sum(y_test==1))

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

classes=['Buche', 'Douglasie', 'Eiche', 'Esche', 'Fichte', 'Kiefer', 'Roteiche']

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index=classes,
                          columns=classes)

figure = plt.figure(figsize=(7, 7))
seaborn.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


