import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage import io
from sklearn.utils import shuffle
from scipy import ndimage

base_path = os.path.dirname(os.getcwd())


def load_data():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    image_id = df[['id']].values
    species = df.species
    species = species.values.reshape((species.shape[0], 1))
    stacked = np.concatenate((image_id, species), axis=1)
    image_id, labels = convert_species_to_labels(stacked)
    images = convert_ids_to_images(image_id)
    more_images, more_labels = augment_data(images, labels)
    return more_images, more_labels


def load_full_data():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    image_id = df[['id']].values
    species = df.species
    species = species.values.reshape((species.shape[0], 1))
    stacked = np.concatenate((image_id, species), axis=1)
    image_id, labels = convert_species_to_labels(stacked)
    images = convert_ids_to_images(image_id)
    return images, labels


def load_test_data():
    df = pd.read_csv('{}/test.csv'.format(base_path))
    image_id = df[['id']].values
    images = convert_ids_to_images(image_id)
    return augment_test_data(images), image_id, convert_labels_to_species()


def augment_data(images, labels):
    more_images = np.zeros((8*images.shape[0], images.shape[1], images.shape[2],))
    more_labels = np.zeros((8*labels.shape[0]))
    for i in range(labels.shape[0]):
        for j in range(8):
            rotation = j*90
            more_images[8*i+j] = ndimage.rotate(images[i], rotation)
            more_labels[8*i+j] = labels[i]
    return shuffle(more_images, more_labels)


def augment_test_data(images):
    more_images = np.zeros((4*images.shape[0], images.shape[1], images.shape[2],))
    for i in range(images.shape[0]):
        for j in range(4):
            rotation = j*90
            more_images[4*i+j] = ndimage.rotate(images[i], rotation)
    return more_images


def convert_species_to_labels(data):
    # create empty array to store new labels in
    labels = np.zeros((data.shape[0],))
    # create empty array to store image ID's in
    image_id = np.zeros((data.shape[0],))
    # find all unique species
    unique_species = np.unique(data[:, 1])
    # label counter for assigning labels
    label_counter = 0
    # assign numberical labels for species
    for species in unique_species:
        ind = np.where(data[:, 1] == species)
        labels[ind] = label_counter
        image_id[ind] = data[ind, 0]
        label_counter += 1
    return image_id, labels


def convert_labels_to_species():
    df = pd.read_csv('{}/train.csv'.format(base_path))
    data = df.species
    # find all unique species
    unique_species = np.unique(data)
    # create empty list to store species
    species_list = []
    # assign species for numberical labels
    for species in unique_species:
        species_list.append(str(species))
    return species_list


def convert_ids_to_images(ids):
    num_ids = ids.shape[0]
    images = np.zeros((num_ids, 32, 32,), dtype=np.uint32)
    for i in range(num_ids):
        images[i] = load_image(ids[i])
    return images


def load_image(image_id):
    return io.imread('{0}/processed/{1}.jpg'.format(base_path, str(int(image_id))))
