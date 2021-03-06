import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import h5py
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=2048, help='number of points to be sampled from cloud')
parser.add_argument('--k_nadel', type=int, default=1, help='number of trees to be sampled per nadel')
parser.add_argument('--k_laub', type=int, default=1, help='number of trees to be sampled per laub')
parser.add_argument('--cutoff', type=float, default=0.0, help='percentage of data to be cut from tree')
parser.add_argument('--species', type=bool, default=True, help='')

FLAGS = parser.parse_args()

NUM_POINTS = FLAGS.num_points
K_NADEL = FLAGS.k_nadel
K_LAUB = FLAGS.k_laub
CUTOFF = FLAGS.cutoff
SPECIES = FLAGS.species



def get_data(directory, k=1, cutoff=CUTOFF, species=SPECIES, n_points=NUM_POINTS):
    label_list = os.listdir(directory)
    trees = []
    tree_labels = []
    group = os.path.split(directory)[1]

    for label in label_list:

        path = os.path.join(directory, label)
        file_list = os.listdir(path)
        print(label)


        for file in file_list:
            file_path = os.path.join(path, file)
            lines = np.loadtxt(file_path)[:, :3]

            if cutoff > 0:
                lines = lines[lines[:, 2].argsort()]
                bound = int(cutoff * lines.shape[0])
                lines = lines[bound:, :]

            if species:
                tree_labels.extend(repeat(label, k))
            else:
                tree_labels.extend(repeat(group, k))

            for i in range(k):
                idx = np.random.randint(lines.shape[0], size=n_points)
                lines_new = lines[idx, :]
                trees.append(lines_new)

        data = np.array(trees)


    return [tree_labels, data]



# import data from files
n_labels, n_trees = get_data('../data/Nadelbäume', n_points=NUM_POINTS, k=K_NADEL)
l_labels, l_trees = get_data('../data/Laubbäume', n_points=NUM_POINTS, k=K_LAUB)


labels = np.append(l_labels, n_labels)

trees = np.concatenate([l_trees, n_trees], axis=0)


## new trees

file_list = os.listdir('../data/add_trees')

for i in range(len(file_list)):
    path = os.path.join('../data/add_trees', file_list[i])
    file_list[i] = file_list[i].replace('XYZ_', '')
    new_path = os.path.join('../data/add_trees', file_list[i])
    os.rename(path, new_path)
    file_list[i] = file_list[i][:-4]

label_map = pd.read_excel('../data/Mappe1.xlsx', header=None)
label_map.columns = ['file_names', 'species']

for i in range(len(label_map.file_names)):
    if label_map.file_names[i][-1] == '_':
        label_map.file_names[i] = label_map.file_names[i][:-1]

latin = np.unique(label_map.species).tolist()
german = ['Spitzahorn', 'Rotbuche', 'Europäische Lerche', 'Holzapfel',
          'Fichte', 'Kiefer', 'Douglasie', 'Traubeneiche', np.NaN]

label_dict = dict(zip(latin, german))

label_dict

label_map.replace({'species': label_dict}, inplace=True)
drop_idx = label_map[label_map['species'] == 'inv_species'].index
label_map.drop(drop_idx, axis=0, inplace=True)

new_trees = []

for index, row in label_map.iterrows():
    if row['file_names'] in file_list:
        path = os.path.join('../data/add_trees', row['file_names'] + '.txt')
        tree = np.genfromtxt(path)
        if tree.shape[0] < 50000:
            continue
        tree = tree[~np.isnan(tree).any(axis=1)]

        if CUTOFF > 0:
            tree = tree[tree[:, 2].argsort()]
            bound = int(CUTOFF * tree.shape[0])
            tree = tree[bound:, :]

        idx = np.random.randint(tree.shape[0], size=NUM_POINTS)
        tree = tree[idx, :]

        trees = np.vstack((trees, tree[None]))
        labels = np.append(labels, row['species'])

for i in range(0,trees.shape[0]):
    fig = plt.figure(figsize=(1.5, 2.25))
    ax = Axes3D(fig)
    plt.axis('off')
    ax.scatter(trees[i,:,0], trees[i,:,1], trees[i,:,2], s=0.1, alpha=1, color='black')
    for ii in range(0, 360, 36):
        ax.view_init(elev=10., azim=ii)
        fig.savefig('Images/' + str(labels[i]) +'/' + '_' + str(i) + '_' + str(labels[i]) + "%d.png" % ii)
        plt.close(fig)