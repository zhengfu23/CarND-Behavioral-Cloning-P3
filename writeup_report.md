** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup_images/center_2017_10_04_10_31_34_712.jpg "Recovery Image"
[image2]: ./writeup_images/center_2017_10_04_10_30_09_100.jpg "Recovery Image 2"
[image3]: ./writeup_images/flipped.jpg "Flipped Image"
[image4]: ./writeup_images/normal.jpg "Normal Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: containing the script to create and train the model
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network
* run1.mp4: containing the recorded run showing autonomous driving footage on track 1.
* writeup_report.md: summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the NVIDIA architecture.

The model includes RELU layers to introduce nonlinearity (code line 58), and the data is normalized in the model using a Keras lambda layer (code line 56).

#### 2. Attempts to reduce overfitting in the model

The model uses l2 regularization to reduce overfitting in all layers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

The model uses a correction parameter to add offset to steering angle assigned to left and right camera images.

I also tuned the train/validation split ratio to reduce overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as driving both clockwise and counterclockwise around the track.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model used by NVIDIA for their self-driving car. I thought this model should be appropriate because it was tested on real world conditions therefore should be more than capable here in the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that the train/validation split is increased from 0.2 to 0.25. I also included regularization into my convolution layers as well as the fully connected layers.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I tried making more simulator runs around the problematic areas and tried additional runs that recovers from side of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-69) consisted of a convolution neural network with the following layers and layer sizes:

* Lambda layer that normalizes the input data to a mean of 0.
* Cropping layer that keeps only the part of the image with the road.
* Convolution layer with 5x5 filter and a depth of 24, with ReLU activation.
* Convolution layer with 5x5 filter and a depth of 36, with ReLU activation.
* Convolution layer with 5x5 filter and a depth of 48, with ReLU activation.
* Convolution layer with 3x3 filter and a depth of 64, with ReLU activation.
* Convolution layer with 3x3 filter and a depth of 64, with ReLU activation.
* Flatten layer.
* Fully connected layer with 100 output, with ReLU activation.
* Fully connected layer with 50 output, with ReLU activation.
* Fully connected layer with 10 output, with ReLU activation.
* Fully connected layer with 1 output.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself back to the center when a lane departure happens.

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would double my data set without recording new driving data.

After the collection process, I had 10386 data points. I then preprocessed this data by normalizing the pixel values to a mean of 0 as well as cropping the part of the images that involve only trees and background as well as the hood of the car.


I finally randomly shuffled the data set and put 25% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by running the training process multiple times and realizes that validation error stops decreasing at the 4th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

##### Recovery from right side
![alt text][image2]

##### Recovery from right side
![alt text][image1]

##### Normal Image
![alt text][image4]

##### Flipped Image
![alt text][image3]

