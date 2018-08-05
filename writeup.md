# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



[track1]: ./examples/track1.jpg "track1"
[track1_counterwise]: ./examples/track1_counterwise.jpg "track1_counterwise"
[track1_recover1]: ./examples/track1_recover1.jpg "track1_recover1"
[track1_recover2]: ./examples/track1_recover2.jpg "track1_recover2"
[track1_recover3]: ./examples/track1_recover3.jpg "track1_recover3"
[track2]: ./examples/track2.jpg "track2"
[track2_counterwise]: ./examples/track2_counterwise.jpg "track2_counterwise"
[track1_flip]: ./examples/track1_flip.jpg "track1_flip"
[track2_flip]: ./examples/track2_flip.jpg "track2_flip"
[track1_shadow]: ./examples/track1_shadow.jpg "track1_shadow"
[track2_shadow]: ./examples/track2_shadow.jpg "track2_shadow"
[track1_crop]: ./examples/track1_crop.jpg "track1_crop"
[track2_crop]: ./examples/track2_crop.jpg "track2_crop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

When I run example code, I get really good result for track1. But I can not let car run on track2 no matter how track2 data quality. The problem is speed. In track1, I can run car smoothly with 30 mph speed. For track2, it's really hard to get that result. Because the steering angle is affected by speed, I add speed as my network input.

In order to judge how well the model was working, I split my image, speed and steering angle data into a training and validation set. 
To combat the overfitting, I modified the model with dropout and batch normalization layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track because shadow. To improve the driving behavior in these cases, I randomly turn down the brightness on images. That is also why I don't convert image to grayscale because shadow will affect angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a convolution neural network with:

| Layer                 |     Description     | 
|:---------------------:|:---------------------------------------------:| 
| ImageInput            | 90x320x3 RGB image | 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 90x320x8 |
| BatchNormalization    ||
| LeakyRELU             ||
| Max pooling	      	| 2x2 stride,  outputs 45x160x8 |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 45x160x16 |
| BatchNormalization    ||
| LeakyRELU             ||
| Max pooling	      	| 2x2 stride,  outputs 23x80x16 |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 23x80x32 |
| BatchNormalization    ||
| LeakyRELU             ||
| Max pooling	      	| 2x2 stride,  outputs 12x40x32 |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x40x32 |
| BatchNormalization    ||
| LeakyRELU             ||
| Max pooling	      	| 2x2 stride,  outputs 6x20x32 |
| Flatten               ||
| Dropout               ||
| Concate SpeedInput    ||
| Fully connected       | 16 |
| BatchNormalization    ||
| LeakyRELU             ||
| Linear Output| 1 |        

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![track1][track1]
![track1_counterwise][track1_counterwise]

These are track2:

![track2][track2]
![track2_counterwise][track2_counterwise]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from sides of road. These images show what a recovery looks like starting from sides:

![track1 recover1][track1_recover1]
![track1 recover2][track1_recover2]
![track1 recover3][track1_recover3]

The first thing I did is cropping image to 90x320 because there is too mush unuse information in image.

![track1_crop][track1_crop]
![track2_crop][track2_crop]

Next, I randomly shuffled the data set and put 20% of the data into a validation set. 

To augment the training dataset, I flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![track1_flip][track1_flip]
![track2_flip][track2_flip]

These are augmented shadow images:

![track1_shadow][track1_shadow]
![track2_shadow][track2_shadow]

I used this augmented training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Testing Result:

I use simulator autonumous mode to test model for track1 with differenct speed: 9, 15, 30 mph. Track2 only test 9 mph because the curve of road is too big. It's hard to drive in high speed.
