from sklearn.cross_validation import train_test_split
import numpy as np
import re
import csv
import os


# We will use 80% for training and 20% for validation
def split_data(x, y):
    data_train, data_valid, labels_train, labels_valid = train_test_split(x, y, test_size=0.20, random_state=42)
    print("Shape of training examples = ", data_train.shape)
    print("Shape of validation examples = ", data_valid.shape)
    return data_train, data_valid, labels_train, labels_valid


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


def reformat(dataset):
    return dataset.reshape((-1, 32, 32, 1)).astype(np.float32)


def average_probs(probs):
    avg_probs = np.zeros((99,))
    for j in range(4):
        avg_probs += probs[j]
    avg_probs = avg_probs/4
    return np.round(avg_probs, 4)


def convert_list_of_ints_to_string(list_of_ints):
    return re.sub('\s+', ',', np.array_str(list_of_ints).strip('[]'))


def convert_array_of_ints_to_string(array_of_ints):
    return np.array_str(array_of_ints).strip('[]')


def scale_probabilities(row):
    return row/(np.max(row) + 1e-10)


def write_results_to_file(species, ids, probs):
    # Make a path for our results to be saved to
    if not os.path.exists('results'):
        os.makedirs('results')
    print('Writing results to file')
    with open('results/results.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
        header = 'id,' + ','.join(species)
        writer.writerow([header])
        for i in range(ids.shape[0]):
            row = average_probs(probs[0][4*i:4*i+4])
            row = convert_list_of_ints_to_string(row)
            row = '{}'.format(str(int(ids[i]))) + row
            writer.writerow([row])
    print('Successfully wrote results to file')
