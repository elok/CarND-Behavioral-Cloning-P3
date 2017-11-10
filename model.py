import csv
import cv2
import os
import numpy as np
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D
from keras.models import Model


def extract_data(data_lines, path):
    orientation_map = {'center': 0, 'left': 1, 'right': 2}

    images = []
    steering_angles = []

    for line in data_lines:
        steering_angle_center = float(line[3])

        STEERING_THRESHOLD = 0.0085

        if abs(steering_angle_center) > STEERING_THRESHOLD:  # Filter by threshold
            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune

            steering_angle_left = steering_angle_center + correction
            steering_angle_right = steering_angle_center - correction

            img_center = process_image(path, line[orientation_map['center']])
            img_left = process_image(path, line[orientation_map['left']])
            img_right = process_image(path, line[orientation_map['right']])

            # add images and angles to data set
            images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_angle_center, steering_angle_left, steering_angle_right])

    # Flip
    augmented_images, augmented_steering_angle_list = [], []
    for image, steering_angle in zip(images, steering_angles):
        augmented_images.append(cv2.flip(image, 1))
        augmented_steering_angle_list.append(steering_angle * -1.0)

    X_train = np.concatenate([images, augmented_images])
    y_train = np.concatenate([steering_angles, augmented_steering_angle_list])

    return X_train, y_train

def process_image(path, file_path):
    """
    Image augmentation
    :param img_data:
    :return:
    """
    filename = file_path.split('/')[-1]
    current_path = os.path.join(path, os.path.join(r'training_data/IMG/', filename))
    image = cv2.imread(current_path)
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def run(location):
    if location == 'home':
        PATH = r'/Users/ericlok/Downloads'
    elif location == 'work':
        PATH = r'C:\Users\elok\Downloads'
    else: # AWS
        PATH = r'/home/carnd/CarND-Behavioral-Cloning-P3'

    lines = []
    with open(os.path.join(PATH, 'training_data/driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    del (lines[0])  # Remove the header (center, left, right, steering, throttle, brake, speed)

    # Extract image and measurements from files
    images, steering_angles = extract_data(lines, PATH)

    X_train = np.array(images)
    y_train = np.array(steering_angles)



    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

    model.save('model.h5')

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [--location home/work/aws] ", version="%prog 1.0")
    parser.add_option("-l", "--location",
                      dest="location",
                      default='aws',
                      help="location")

    (options, args) = parser.parse_args()

    run(location=options.location)