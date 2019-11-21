import numpy as np
import os
import random


def get_data(directory):
    label_list = os.listdir(directory)
    trees = []
    counts = []

    for label in label_list:

        path = os.path.join(directory, label)
        file_list = os.listdir(path)
        counts.append(len(file_list))

        for file in file_list:
            file_path = os.path.join(path, file)

            with open(file_path, 'r+') as f:
                lines = [line.rstrip() for line in f]

            for i in range(0, len(lines)):
                lines[i] = [float(x) for x in lines[i].split()]

            lines = random.sample(lines, 2048)
            lines = np.array(lines)[:,:3]
            trees.append(lines)

        data = np.array(trees)

    return [data, counts]


# import data from files
n_trees, n_counts = get_data('../data/Nadelbäume')
l_trees, l_counts = get_data('../data/Laubbäume')











    




