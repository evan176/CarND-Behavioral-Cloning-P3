import csv
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras import regularizers, optimizers
from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, concatenate


images = []
speeds = []
angles = []
throttles = []


def get_data(path, images, speeds, angles, throttles):
    lines = []
    with open(os.path.join(path, "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(path, "IMG", filename)
        img = cv2.imread(current_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[50:-20, :, :]
        images.append(img)
        angles.append([float(line[3])])
        throttles.append([float(line[4])])
        speeds.append([float(line[6]) / 30.0])
    return images, speeds, angles, throttles


images, speeds, angles, throttles = get_data("track1_clockwise_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track1_counterclockwise_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track1_clockwise_various_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track1_counterclockwise_various_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track1_recover_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track2_clockwise_various_data", images, speeds, angles, throttles)
images, speeds, angles, throttles = get_data("track2_counterclockwise_various_data", images, speeds, angles, throttles)


def generator(sample_indexes, batch_size=32, is_train=False):
    augments = []
    for ind in sample_indexes:
        if is_train:
            augments.append((ind, -1, 1))
            augments.append((ind, -1, 0))
            augments.append((ind, 1, 1))
        augments.append((ind, 1, 0))

    num_indexes = len(augments)
    while 1:
        shuffle = random.sample(augments, num_indexes)
        for offset in range(0, num_indexes, batch_size):
            samples = shuffle[offset:offset + batch_size]
            sample_images = []
            sample_speeds = []
            sample_angles = []
            for aug in samples:
                shadow = random.randrange(10, 100)
                if aug[2] == 1:
                    p1, p2 = random.sample(range(4), 2)
                    def generate_point(p):
                        if p == 0:
                            return random.randint(0, 320), 0
                        elif p == 1:
                            return 160, random.randint(0, 160)
                        elif p == 2:
                            return random.randint(0, 320), 160
                        elif p == 3:
                            return 0, random.randint(0, 160)

                    x1, y1 = generate_point(p1)
                    x2, y2 = generate_point(p2)

                    X_m = np.mgrid[0:height, 0:width][1]
                    Y_m = np.mgrid[0:height, 0:width][0]

                    img = images[aug[0]][:, ::aug[1], :].astype(np.int16)
                    if random.random() > 0.5:
                        img[(x2 - x1) * (Y_m - y1) - (y2 - y1) * (X_m - x1) > 0] -= shadow
                    else:
                        img[(x2 - x1) * (Y_m - y1) - (y2 - y1) * (X_m - x1) < 0] -= shadow
                    img[img < 0] = 0
                else:
                    img = images[aug[0]][:, ::aug[1], :]

                sample_images.append(img)

                sample_speeds.append(speeds[aug[0]])

                if aug[1] == 1:
                    sample_angles.append(angles[aug[0]])
                else:
                    sample_angles.append([-angles[aug[0]][0]])

            yield [np.array(sample_images), np.array(sample_speeds)], np.array(sample_angles)

indexes = [i for i in range(len(images))]
valid_indexes = random.sample(indexes, int(len(images) * 0.2))
train_indexes = [ind for ind in indexes if ind not in valid_indexes]
train_generator = generator(train_indexes, is_train=True)
validation_generator = generator(valid_indexes)

height, width, input_channels = images[0].shape
l2_rate = 0

img_input = Input(shape=(height, width, input_channels), dtype='int32')
speed_input = Input(shape=(1,), dtype='float32')

norm_img = Lambda(lambda x: np.divide(x, 255) - 0.5)(img_input)

conv1 = Conv2D(
        filters=8,
        kernel_size=(5, 5),
        padding='same',
        activation='linear',
        kernel_regularizer=regularizers.l2(l2_rate)
        )(norm_img)
norm_conv1 = BatchNormalization()(conv1)
active_conv1 = LeakyReLU()(norm_conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(active_conv1)

conv2 = Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='same',
        activation='linear',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(pool1)
norm_conv2 = BatchNormalization()(conv2)
active_conv2 = LeakyReLU()(norm_conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(active_conv2)

conv3 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='linear',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(pool2)
norm_conv3 = BatchNormalization()(conv3)
active_conv3 = LeakyReLU()(norm_conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(active_conv3)

conv4 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='linear',
        padding='same',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(pool3)
norm_conv4 = BatchNormalization()(conv4)
active_conv4 = LeakyReLU()(norm_conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(active_conv4)

flat = Flatten()(pool4)
drop_flat = Dropout(0.5)(flat)
cat_flat_speed = concatenate([drop_flat, speed_input])

fully1 = Dense(16, kernel_regularizer=regularizers.l2(l2_rate))(cat_flat_speed)
norm_fully1 = BatchNormalization()(fully1)
active_fully1 = LeakyReLU()(norm_fully1)

drop_fully1 = Dropout(0.5)(active_fully1)
output = Dense(1, kernel_regularizer=regularizers.l2(l2_rate))(drop_fully1)

model = Model(inputs=[img_input, speed_input], outputs=[output])

model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-3))
model.fit_generator(train_generator, steps_per_epoch=len(train_indexes) / 32 * 4, validation_data=validation_generator, validation_steps=len(valid_indexes) / 32, nb_epoch=10)
model.save('model.h5')
