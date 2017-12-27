'''
Leaf Classifier via Deep Convolutional Neural Network

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse
import tensorflow.contrib.slim as slim

from model_helpers import *
from data_helpers import *
from network import *

# Setting the training parameters

# Number of possible actions
actions = 99
# How many experience traces to use for each training step.
batch_size = 64
# Number of training steps
num_steps = 5501

# The path to save our model to.
path = "./cnn"


def train(load_model=False):
    # load data
    images, labels = load_data()

    # convert to training and validation sets
    x_train, x_valid, train_labels, valid_labels = split_data(images, labels)
    train_dataset = reformat(x_train)
    valid_dataset = reformat(x_valid)

    tf.reset_default_graph()
    mainN = Network(is_training=True)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=5)

    # Make list to store losses, accuracies
    losses = []
    accuracies = []
    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Successfully Loaded')
        else:
            sess.run(init)

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size)]
            _, lossA, yP, LO = sess.run([mainN.update, mainN.loss, mainN.probs, mainN.label_oh],
                feed_dict={mainN.data: batch_data, mainN.labels: batch_labels})
            losses.append(lossA)
            accuracies.append(accuracy(yP, LO))
            if (step % 100 == 0):
                print('Minibatch loss at step %d: %f' % (step, lossA))
                print('Minibatch accuracy: %.1f%%' % accuracy(yP, LO))
                yP, LO = sess.run([mainN.probs, mainN.label_oh],
                    feed_dict={mainN.data: valid_dataset, mainN.labels: valid_labels})
                print('Validation accuracy: %.1f%%' % accuracy(yP, LO))
                saver.save(sess, path+'/model-'+str(step)+'.cptk')
                print("Saved Model")
        yP, LO = sess.run([mainN.probs, mainN.label_oh],
            feed_dict={mainN.data: valid_dataset, mainN.labels: valid_labels})
        print('Validation accuracy: %.1f%%' % accuracy(yP, LO))
        saver.save(sess, path+'/model-'+str(step)+'.cptk')
        print("Saved Model")
        plt.figure(1)
        plt.title('Training Loss')
        plt.plot(range(len(losses)), losses)
        plt.figure(2)
        plt.title('Training Accuracies')
        plt.plot(range(len(accuracies)), accuracies)
        plt.show()


def validate():
    tf.reset_default_graph()
    mainN = Network(is_training=False)

    saver = tf.train.Saver(max_to_keep=5)

    # load data
    images, labels = load_data()

    # convert to training and validation sets
    x_train, x_valid, train_labels, valid_labels = split_data(images, labels)
    valid_dataset = reformat(x_valid)

    with tf.Session() as sess:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Loaded!')

        yP, LO = sess.run([mainN.probs, mainN.label_oh],
            feed_dict={mainN.data: valid_dataset, mainN.labels: valid_labels})
        print('Validation accuracy: %.1f%%' % accuracy(yP, LO))


def test():
    tf.reset_default_graph()
    mainN = Network(is_training=False)

    saver = tf.train.Saver(max_to_keep=5)

    # load data
    images, image_id, species = load_test_data()

    test_dataset = reformat(images)
    test_dataset.astype(float)
    with tf.Session() as sess:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Loaded!')

        yP = sess.run([mainN.probs], feed_dict={mainN.data: test_dataset})
        np.save('testProbs', yP)
        print('Completed processing {} test images'.format(str(image_id.shape[0])))
        write_results_to_file(species, image_id, yP)


def writeResults():
    # load data
    images, image_id, species = load_test_data()
    # load saved results
    probs = np.load('testProbs.npy')
    write_results_to_file(species, image_id, probs)


def main():
    parser = argparse.ArgumentParser(description="Train or run leaf classifier")
    parser.add_argument("-m", "--mode", help="Train / Run / Validate", required=True)
    parser.add_argument("-l", "--load", help="Load previously trained weights? True/False")
    args = vars(parser.parse_args())
    if args['mode'] == 'Train':
        if args['load']:
            if args['load'] == 'True':
                train(load_model=True)
        else:
            train(load_model=False)
    elif args['mode'] == 'Test':
        test()
    elif args['mode'] == 'Validate':
        validate()
    elif args['mode'] == 'Write':
        writeResults()
    else:
        print(':p Invalid Mode.')


if __name__ == "__main__":
    main()
