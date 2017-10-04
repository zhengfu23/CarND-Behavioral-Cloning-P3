import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = [0.0, 0.5, -0.5]
for line in lines:
    for i in range(3):
        src_path = line[i]
        filename = src_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        img = cv2.imread(current_path)
        img_flipped = np.fliplr(img)
        images.append(img)
        images.append(img_flipped)
        measurement = float(line[3]) + correction[i]
        measurements.append(measurement)
        measurements.append(-measurement)


X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, dilation_rate=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, 5, dilation_rate=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, 5, dilation_rate=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, 3, padding='valid', activation='relu'))
model.add(Conv2D(64, 3, padding='valid', activation='relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
