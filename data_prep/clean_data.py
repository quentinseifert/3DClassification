import os

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

label_list = os.listdir('../data/')

for label in label_list:
    species_list = os.listdir('../data/' + label)
    for species in species_list:
        path = os.path.join('../data', label, species)
        clean_trees(path)





