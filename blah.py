import csv
import cv2
import os
import numpy as np
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def extract_list(data_lines, path, orientation_list):
    image_list = []
    measurement_list = []

    for orient in orientation_list:
        images_curr, measurements_curr = extract(data_lines=data_lines, orientation=orient, path=path)

        image_list.append(images_curr)
        measurement_list.append(measurements_curr)

    images = np.concatenate(image_list)
    measurements = np.concatenate(measurement_list)
    return images, measurements

def extract(data_lines, orientation, path):
    orientation_map = {'center': 0, 'left': 1, 'right': 2}

    images = []
    steering_angles = []

    for line in data_lines:
        source_path = line[orientation_map[orientation]]
        filename = source_path.split('/')[-1]
        current_path = os.path.join(path, os.path.join(r'training_data/IMG/', filename))
        image = cv2.imread(current_path)
        images.append(image)
        steering_angle_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        if orientation == 'left':
            steering_angle_center += correction
        elif orientation == 'right':
            steering_angle_center -= correction

        steering_angles.append(steering_angle_center)

    return images, steering_angles

def get_pipeline(crop=False):
    model = Sequential()
    # Cropping
    if crop:
        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))
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
    return model

def get_nvidia_pipeline(crop=False):
    model = Sequential()
    # Cropping
    if crop:
        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))
        # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # Normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def run(location):
    if location == 'home':
        PATH = r'/Users/ericlok/Downloads'
    elif location == 'work':
        PATH = r'C:\Users\elok\Downloads'
    else: # AWS
        PATH = r'/home/carnd/CarND-Behavioral-Cloning-P3'

    lines = []
    with open(os.path.join(PATH, r'training_data/driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    del (lines[0]) # Remove the header (center, left, right, steering, throttle, brake, speed)

    # Extract image and measurements from files
    images, measurements = extract_list(lines, PATH, ['center'])

    # Flip
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

    model = get_pipeline(crop=False)
    # model = get_nvidia_pipeline(crop=False)

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