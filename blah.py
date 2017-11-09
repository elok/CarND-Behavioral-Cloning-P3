import csv
import cv2
import os
import numpy as np
from optparse import OptionParser


def run(location):
    if location == 'home':
        PATH = r'/Users/ericlok/Downloads'
    elif location == 'work':
        PATH = r'C:\Users\elok\Downloads'
    else:  # AWS
        PATH = r'/home/carnd/CarND-Behavioral-Cloning-P3'

    lines = []
    with open(os.path.join(PATH, 'training_data/driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(PATH,
                                    'training_data/IMG/' + filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    # flip
    # augmented_images, augmented_measurements = [], []
    # for image, measurement in zip(images, measurements):
    #     augmented_images.append(image)
    #     augmented_measurements.append(measurement)
    #     augmented_images.append(cv2.flip(image,1))
    #     augmented_measurements.append(measurement*-1.0)

    X_train = np.array(images)
    y_train = np.array(measurements)

    # X_train = np.concatenate([images, augmented_images])
    # y_train = np.concatenate([measurements, augmented_measurements])

    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Cropping2D
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    model = Sequential()
    # Cropping
    # if crop:
    #     model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))
    # Normalize
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # 1
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    # 2
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    # 3
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save('model.h5')

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [--location home/work/aws] ", version="%prog 1.0")
    parser.add_option("-l", "--location",
                      dest="location",
                      default='aws',
                      help="location")

    (options, args) = parser.parse_args()
    run(location=options.location)