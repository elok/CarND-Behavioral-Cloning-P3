# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[driving_forward]: ./examples/driving_forward.jpg "driving_forward"
[driving_backwards]: ./examples/driving_backwards.jpg "driving_backwards"
[recovery_left]: ./examples/recovery_left.jpg "recovery_left"
[recovery_right]: ./examples/recovery_right.jpg "recovery_right"
[driving_forward_left]: ./examples/driving_forward_left.jpg "driving_forward_left"
[driving_forward_right]: ./examples/driving_forward_right.jpg "driving_forward_right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I first used the default model from the lesson. After adding in the normalization, augmentation, and extra datasets, the results were not bad but could use some improvement. After reading up some more on the nvdia architecture, I decided to use it and it gave me better results.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 134).
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving backwards on the track. 

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a model that was tried and true and copy off of that.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because NVIDIA used it successfully on actual self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added a dropout layer to the model. 

Then I flipped the images horizontally and added more data by driving track one backwards.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I had a "correction" dataset in which I intentionally veered off the track and corrected it by driving it back into the middle. I did this for both the left and right side of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I used the NVIDIA architecture which consists of the following:

* The image is normalized using a Keras lambda layer (line 121). The input is 160 x 320 with 3 dimensions.
* The image is then cropped at the top by 70 and bottom by 25 (line 123)
* Apply a 5x5 convolution with 24 output filters, 2x2 stride, and relu activation (line 125)
* Apply a 5x5 convolution with 36 output filters, 2x2 stride, and relu activation
* Apply a 5x5 convolution with 48 output filters, 2x2 stride, and relu activation
* Apply a 3x3 convolution with 64 output filters, and relu activation
* Apply a 3x3 convolution with 64 output filters, and relu activation
* Dropout with probability of 0.5
* Flatten
* Dense 100
* Dense 50
* Dense 10
* Dense 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][driving_forward]

I then recorded a lap driving backwards on the track using the center lane driving. Here is an example image of driving backwards:

![alt text][driving_backwards]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back into the center on its own. These images show what a recovery looks like starting from the extreme left and right:

![alt text][recovery_left]
![alt text][recovery_right]

To augment the data set, I also flipped images horizontally and cropped the upper portion of the images.

After the collection process, I had 13,068 number of data points. I then preprocessed this data by normalizing it and cropping the upper portion.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the training loss falling below 0.01 I used an adam optimizer so that manually training the learning rate wasn't necessary.
