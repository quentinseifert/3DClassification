import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=2048, help='number of points to be sampled from cloud')
parser.add_argument('--k_nadel', type=int, default=1, help='number of trees to be sampled per nadel')
parser.add_argument('--k_laub', type=int, default=1, help='number of trees to be sampled per laub')
parser.add_argument('--cutoff', type=float, default=0.0, help='percentage of data to be cut from tree')
parser.add_argument('--species', type=bool, default=True, help='sort categories by species')
FLAGS = parser.parse_args()

NUM_POINTS = FLAGS.num_points
K_NADEL = FLAGS.k_nadel
K_LAUB = FLAGS.k_laub
CUTOFF = FLAGS.cutoff
SPECIES = FLAGS.species

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
n_trees, n_counts = get_data('../data/Nadelbäume', n_points=6000, k=5)
l_trees, l_counts = get_data('../data/Laubbäume', n_points=6000, k=2)


for i in range (0,l_trees.shape[0]):
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    plt.axis('off')
    ax.scatter(l_trees[i,:,0], l_trees[i,:,1], l_trees[i,:,2], s=20, alpha=1, color='black')
    fig.savefig('Images/laub/' + 'laub' + str(i) +'.jpg')
    plt.close(fig)

for j in range(0, l_trees.shape[0]):
    fig1 = plt.figure()
    ax1 = plt.subplot(111, projection='3d')
    plt.axis('off')
    ax1.scatter(l_trees[i,:,0], l_trees[j,:,1], l_trees[j,:,2], s=20, alpha=1, color='black')
    fig1.savefig('Images/laub/' + 'laub' + str(j) + '.' + '1.jpg')
    plt.close(fig1)

for i in range (0,n_trees.shape[0]):
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    plt.axis('off')
    ax.scatter(n_trees[i,:,0], n_trees[i,:,1], n_trees[i,:,2], s=20, alpha=1, color='black')
    fig.savefig('Images/nadel/' + 'nadel' + str(i) +'.jpg')
    plt.close(fig)

for j in range(0, n_trees.shape[0]):
    fig1 = plt.figure()
    ax1 = plt.subplot(111, projection='3d')
    plt.axis('off')
    ax1.scatter(n_trees[j,:,0], n_trees[j,:,1], n_trees[j,:,2], s=20, alpha=1, color='black')
    fig1.savefig('Images/nadel/' + 'nadel' + str(j) + '.' + '1.jpg')
    plt.close(fig1)





