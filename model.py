import csv
import cv2
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
import imageio
from sklearn.model_selection import train_test_split

import numpy as np
import sklearn

data_dir = './driving_data/'


# Read the samples from specific data directory such as ./driving_data/driving_data_track1_forward/
def read_samples(samples, dir, skip_header=False):
    with open(data_dir + dir + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        if skip_header:
            next(reader)
        for line in reader:
            samples.append((line, dir))


samples = []

def read_all_samples():
    read_samples(samples, 'driving_data_udacity', skip_header=True)
    read_samples(samples, 'driving_data_track1_forward')
    read_samples(samples, 'driving_data_track1_backward')
    read_samples(samples, 'driving_data_track1_forward_recovery')

    read_samples(samples, 'driving_data_track2_forward')
    read_samples(samples, 'driving_data_track2_backward')

def get_sample_path(data_dir, image_dir, sample_dir):
    return data_dir + image_dir + '/IMG/' + sample_dir.split('/')[-1]

# Generate the augmented image data.
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                sample, imagedir = batch_sample
                name = get_sample_path(data_dir, imagedir, sample[0])
                center_image = imageio.imread(name)
                center_angle = float(sample[3])

                images.append(center_image)
                angles.append(center_angle)

                # Augment the center image by flipping it and negating the angle.
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)

                correction = 0.2
                center_angle_left = center_angle + correction
                name_left = get_sample_path(data_dir, imagedir, sample[1])
                image_left = imageio.imread(name_left)

                center_angle_right = center_angle - correction
                name_right = get_sample_path(data_dir, imagedir, sample[2])
                image_right = imageio.imread(name_right)

                images.append(image_left)
                angles.append(center_angle_left)

                # Augment the left camera image by flipping it and negating the angle.
                images.append(cv2.flip(image_left, 1))
                angles.append(-center_angle_left)

                images.append(image_right)
                angles.append(center_angle_right)

                # Augment the right camera image by flipping it and negating the angle.
                images.append(cv2.flip(image_right, 1))
                angles.append(-center_angle_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size = 32

read_all_samples()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320


# NVIDIA PilotNet as per https://arxiv.org/pdf/1704.07911.pdf
def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


model = get_model()

steps_per_epoch = math.ceil(len(train_samples) / batch_size)
validation_steps = math.ceil(len(validation_samples) / batch_size)

# Train for 6 epochs.
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
                    validation_steps=validation_steps, epochs=6, verbose=1)

model.save("./model.h5")
