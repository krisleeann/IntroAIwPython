import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import Conv2D

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Access categories and loop through
    # Slight bug, changed from os.listdir(data_dir) because of runtime issue
    for i in range(NUM_CATEGORIES):
        path = os.listdir(os.path.join(data_dir, str(i)))

        for p in path:
            # Read and resize image
            img_read = cv2.imread(os.path.join(data_dir, str(i), str(p)))
            img_resize = cv2.resize(img_read, (IMG_WIDTH, IMG_HEIGHT))

            # Append images and labels variables to respective their lists
            images.append(img_resize)
            labels.append(int(i))

    # Specs: return tuple 'images, labels'
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Code adapted from: https://machinelearningmastery.com/keras-functional-api-deep-learning/

    # `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`
    visible = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    conv1 = Conv2D(16, (3, 3), activation="relu")(visible)      # Tried 32, 64 - 0.9533 (runtime = LONG)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation="relu")(pool1)        # Tried 32, 16, 30 - nominal difference
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(pool2)
    hidden1 = Dense(100, activation="relu")(flat)

    # Goal was to keep time per step under double digits (slower comp, 2017)
    # The output layer should have `NUM_CATEGORIES` units (43)
    output = Dense(NUM_CATEGORIES, activation="softmax")(hidden1)
    model = Model(inputs=visible, outputs=output)
    # Dropout rate of 0.5 to avoid overfitting
    Dropout(0.5)

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
