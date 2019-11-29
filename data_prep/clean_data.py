import os
import numpy as np


# define function to remove and replace unwanted signs in files
def clean_trees(path):
    file_list = os.listdir(path)

    for file in file_list:
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            filedata = f.read()

        filedata = filedata.replace(';', ' ')
        filedata = filedata.replace(',', ' ')

        with open(file_path, 'w') as f:
            f.write(filedata)


# use clean trees to go through data and clean all files (actually unnecessary to go through every directory)

label_list = ['Laubbäume', 'Nadelbäume']

for label in label_list:
    species_list = os.listdir('../data/' + label)
    for species in species_list:
        path = os.path.join('../data', label, species)
        clean_trees(path)

file_list = os.listdir('../data/laubbäume/unsortiert')
file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))
path = '../data/laubbäume/unsortiert'
out = []
for file in file_list:
    file_path = os.path.join(path, file)
    print(file)
    with open(file_path, 'r') as f:
        data = [line.rstrip() for line in f]
    if len(data) < 2048:
        out.append(file_path)
        os.remove(file_path)
        continue
    lengths = []
    for i in range(0, len(data)):
        data[i] = [x for x in data[i].split()]
        lengths.append(len(data[i]) < 3)

    idx = [i for i, x in enumerate(lengths) if x]

    for id in sorted(idx, reverse=True):
        del (data[id])

    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]

    np.savetxt(file_path, np.array(data)[:, :3])


# save file names of files that are thrown out
if not os.path.exists('../data/unsorted_labels'):
    os.mkdir('../data/unsorted_labels')

with open('../data/unsorted_labels/out_files.txt', 'w+') as f:
    f.write('\n'.join(idx))








