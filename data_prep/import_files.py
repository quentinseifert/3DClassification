import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=2048, help='number of points to be sampled from cloud')
parser.add_argument('--k_nadel', type=int, default=1, help='number of trees to be sampled per nadel')
parser.add_argument('--k_laub', type=int, default=1, help='number of trees to be sampled per laub')
parser.add_argument('--cutoff', type=float, default=0.0, help='percentage of data to be cut from tree')
FLAGS = parser.parse_args()

NUM_POINTS = FLAGS.num_points
K_NADEL = FLAGS.k_nadel
K_LAUB = FLAGS.k_laub
CUTOFF = FLAGS.cutoff



def get_data(directory, n_points=NUM_POINTS, k=1, cutoff=CUTOFF):
    label_list = os.listdir(directory)
    trees = []
    counts = []

    for label in label_list:

        path = os.path.join(directory, label)
        file_list = os.listdir(path)
        counts.append(len(file_list) * k)
        print(label)

        for file in file_list:
            file_path = os.path.join(path, file)
            lines = np.loadtxt(file_path)[:, :3]

            if (cutoff > 0):
                lines = lines[lines[:, 2].argsort()]
                bound = int(cutoff * lines.shape[0])
                lines = lines[bound:, :]

            for i in range(k):
                idx = np.random.randint(lines.shape[0], size=n_points)
                lines_new = lines[idx, :]
                trees.append(lines_new)

        data = np.array(trees)

    return [data, counts]


# import data from files
n_trees, n_counts = get_data('../data/Nadelbäume', n_points=NUM_POINTS, k=K_NADEL)
l_trees, l_counts = get_data('../data/Laubbäume', n_points=NUM_POINTS, k=K_LAUB)


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
