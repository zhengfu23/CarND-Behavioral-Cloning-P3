# Author: Zheng Fu

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

# Read the log to store the file names and steering angle data to a list
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# Using python generator can avoid high memory usage with large amount of data.
# Instead the training data is generated on-the-fly.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # This correction array is defined to utilize the left and right camera image.
    # This parameter was manually tuned to be the optimal value.
    correction = [0.0, 0.25, -0.25]
    while 1:
        # Shuffling the filenames before making the data set.
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
                    # The images are flipped and negated steering angle added in order to augment the data set
                    img_flipped = np.fliplr(img)
                    images.append(img)
                    images.append(img_flipped)

                    # This is where correction is added to adjust for left or right camera images.
                    measurement = float(batch_sample[3]) + correction[i]
                    measurements.append(measurement)    # adding the original image
                    measurements.append(-measurement)   # adding the flipped image

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Using sklearn's train_test_split to divide the data into train and validation sets.
train_samples, valid_samples = train_test_split(samples, test_size=0.25)

# Define the generators for Keras.
train_generator = generator(train_samples, batch_size=32)
valid_generator = generator(valid_samples, batch_size=32)


# Below is a sequential model based on NVIDIA architecture implemented using Keras framework.
model = Sequential()

# The lambda layer is used to normalize the pixel data to have 0 mean.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# Then Keras' cropping2d layer crops the images so that only the parts important to the model is kept.
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Followed by 4 convolutional layers with different kernel size and depth with relu activation and l2 regularizers.
model.add(Conv2D(24, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(36, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(48, 5, dilation_rate=(2, 2), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(64, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001)))

# Then the output is flatten and connected to three fully connected layers with l2 regularizers.
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

# Last the model produces one output, the predicted steering angle.
model.add(Dense(1))

# The model is trained using mean square error for the lost function since the problem here is similar to linear regression.
# Adam optimizer is used so learning-rate is not needed to be specified.
model.compile(loss='mse', optimizer='adam')

# Last step Keras trains the model using the two data sets generated above
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=4, validation_data=valid_generator, validation_steps=len(valid_samples))

# Saved the model in a h5py format in order to power the autonomous mode in the simulator.
model.save('model.h5')
