import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import h5py


def get_data(directory, n_points=2048, k=1):
    label_list = os.listdir(directory)
    trees = []
    counts = []

    for label in label_list:

        path = os.path.join(directory, label)
        file_list = os.listdir(path)
        counts.append(len(file_list) * k)

        for file in file_list:
            file_path = os.path.join(path, file)

            with open(file_path, 'r+') as f:
                lines = [line.rstrip() for line in f]

            for i in range(0, len(lines)):
                lines[i] = [float(x) for x in lines[i].split()]
            
            
            for i in range(k):
                lines_new = random.sample(lines, n_points)
                lines_new = np.array(lines_new)[:, :3]
                trees.append(lines_new)

        data = np.array(trees)

    return [data, counts]


# import data from files
n_trees, n_counts = get_data('../data/Nadelbäume', n_points=4096,k=3)
l_trees, l_counts = get_data('../data/Laubbäume', n_points=4096)


label_n = np.repeat("Nadel", np.sum(n_counts))
label_l = np.repeat("Laub", np.sum(l_counts))
labels = np.append(label_l, label_n)


trees = np.concatenate([l_trees, n_trees], axis=0)

lab_bin = LabelBinarizer()
labels = lab_bin.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(trees, labels, test_size=0.3, random_state=41)

if not os.path.exists('../h5_data/'):
    os.mkdir('../h5_data/')

with h5py.File('../h5_data/train.h5', 'w') as f:
    f.create_dataset('data', data=x_train)
    f.create_dataset('label', data=y_train)

with h5py.File('../h5_data/test.h5', 'w') as f:
    f.create_dataset('data', data=x_test)
    f.create_dataset('label', data=y_test)
