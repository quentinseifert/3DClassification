import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat

label_map = pd.read_excel('../data/Baumarten_Ordner_unsortiert.xlsx', header=None)
file_list = os.listdir('../data/Laubbäume/Unsortiert')
label_map.columns = ['file_names', 'species']
trees = []
labels = []

species = label_map['species'].unique()
for spec in species:
    if not os.path.exists('Images/' + spec):
        os.mkdir('Images/' + spec)

for i in range(len(file_list)):
    file_list[i] = file_list[i][:-4]

for index, row in label_map.iterrows():
    if str(row['file_names']) in file_list:
        path = os.path.join('../data/Laubbäume/unsortiert', str(row['file_names']) + '.pts')
        with open(path, 'r') as f:
            data = [line.rstrip() for line in f]

        lengths = []
        for i in range(0, len(data)):
            data[i] = [x for x in data[i].split()]
            lengths.append(len(data[i]) < 3)

        idx = [i for i, x in enumerate(lengths) if x]

        for id in sorted(idx, reverse=True):
            del (data[id])

        for i in range(0, len(data)):
            data[i] = [float(x) for x in data[i]]

        np.savetxt(path, np.array(data)[:, :3])

        tree = np.genfromtxt(path)
        tree = tree[~np.isnan(tree).any(axis=1)]

        if tree.shape[0] < 50000:
            continue

        idx = np.random.randint(tree.shape[0], size=6000)
        tree = tree[idx, :]

        trees.append(tree)
        labels.append(row['species'])

trees = np.array(trees)

for i in range(0,trees.shape[0]):
    fig = plt.figure(figsize=(1.5, 2.25))
    ax = Axes3D(fig)
    plt.axis('off')
    ax.scatter(trees[i,:,0], trees[i,:,1], trees[i,:,2], s=0.1, alpha=1, color='black')
    for ii in range(0, 360, 36):
        ax.view_init(elev=10., azim=ii)
        fig.savefig('Images/' + str(labels[i]) +'/' + '_' + str(i) + '_' + str(labels[i]) + "%d.png" % ii)
        plt.close(fig)