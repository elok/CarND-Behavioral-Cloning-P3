import csv
import cv2
import os
import numpy as np
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

def setup_data(data_lines, path, data_group_folder):
    """
    Given the location of the files and driving log raw data, retrieve the images and process them
    :param data_lines: raw data from the driving log
    :param path: root path of the images
    :param data_group_folder: training_data, training_data_backwards, training_data_correction
    :return:
    """
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

            img_center = process_image(path, data_group_folder, line[orientation_map['center']])
            img_left = process_image(path, data_group_folder, line[orientation_map['left']])
            img_right = process_image(path, data_group_folder, line[orientation_map['right']])

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

    print('{0}: # of data lines: {1}, # of items after left and right: {2}, # after flipping: {3}'.format(
                                                                                        data_group_folder,
                                                                                        len(data_lines),
                                                                                        len(images),
                                                                                        len(X_train)))

    return X_train, y_train

def process_image(path, data_group_folder, file_path):
    """
    Given the path of the file, apply any necessary conversions to the image data
    :param path: root path of the image data
    :param data_group_folder: training_data, training_data_backwards, training_data_correction
    :param file_path: path of the file
    :return:
    """
    filename = file_path.split('/')[-1]
    current_path = os.path.join(path, os.path.join(r'{0}/IMG/'.format(data_group_folder), filename))
    image = cv2.imread(current_path)
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def run(location):
    # ---------------------------------------
    # Setup paths
    # ---------------------------------------
    if location == 'home':
        PATH = r'/Users/ericlok/Downloads'
    elif location == 'work':
        PATH = r'C:\Users\elok\Downloads'
    else: # AWS
        PATH = r'/home/carnd/CarND-Behavioral-Cloning-P3'

    groups_of_data = ['training_data', 'training_data_backwards', 'training_data_correction']
    image_list = []
    steering_angle_list = []

    for data_group_folder in groups_of_data:

        lines = []
        with open(os.path.join(PATH, '{0}/driving_log.csv'.format(data_group_folder))) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        print('{0}: number of raw data: {1}'.format(data_group_folder, len(lines)))

        if location == 'work':
            del (lines[0])  # Remove the header (center, left, right, steering, throttle, brake, speed)

        # ---------------------------------------
        # Extract image and measurements from files
        # ---------------------------------------
        images, steering_angles = setup_data(lines, PATH, data_group_folder)
        image_list.append(images)
        steering_angle_list.append(steering_angles)

        print('{0}: number of raw data after processing: {1}'.format(data_group_folder, len(images)))

    X_train = np.concatenate(image_list)
    y_train = np.concatenate(steering_angle_list)
    print('Total number of data points: {0}'.format(len(X_train)))

    # ---------------------------------------
    # Construct the Model
    # ---------------------------------------
    model = Sequential()
    # Normalize the data. Input is 160 x 320 with 3 dimensions.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # Crop the top by 70 and bottom by 25. (begin width, end width) (begin height, end height)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # Apply a 5x5 convolution with 24 output filters, 2x2 stride, and relu activation
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), activation='relu'))
    # Apply a 5x5 convolution with 36 output filters, 2x2 stride, and relu activation
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), activation='relu'))
    # Apply a 5x5 convolution with 48 output filters, 2x2 stride, and relu activation
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), activation='relu'))
    # Apply a 3x3 convolution with 64 output filters, and relu activation
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
    # Apply a 3x3 convolution with 64 output filters, and relu activation
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
    model.add(Dropout(p=0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=100))
    model.add(Dense(output_dim=50))
    model.add(Dense(output_dim=10))
    model.add(Dense(output_dim=1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=15)
    model.save('model.h5')

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [--location home/work/aws] ", version="%prog 1.0")
    parser.add_option("-l", "--location",
                      dest="location",
                      default='aws',
                      help="location")

    (options, args) = parser.parse_args()

    run(location=options.location)