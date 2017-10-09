import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D
from keras import regularizers
from sklearn.model_selection import train_test_split


samples = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = [0.0, 0.25, -0.25]
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    src_path = batch_sample[i]
                    filename = src_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    img = cv2.imread(current_path)
                    img_flipped = np.fliplr(img)
                    images.append(img)
                    images.append(img_flipped)

                    measurement = float(batch_sample[3]) + correction[i]
                    measurements.append(measurement)
                    measurements.append(-measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, valid_samples = train_test_split(samples, test_size=0.25)

train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(valid_samples, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(36, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(48, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=4, validation_data=valid_generator, validation_steps=len(valid_samples))


model.save('model.h5')
