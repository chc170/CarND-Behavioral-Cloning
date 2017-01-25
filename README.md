# Behavioral Cloning Project

In this project a convolutional neural network is built for steering a car in a simulator. The network model is trained on the images collected from the cameras on the car while we are driving the car. After training the model will be able to predict steering angles and drive by itself like a human does.


## Model Architecture
---

There are serveral popular models that has been built for the exact task, for example [Nvidia's](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and [CommaAI's](https://github.com/commaai/research/blob/master/train_steering_model.py) models. In this project Nvidia's model, CommaAI's model, and my own customized model are used for experiment and the final result came from Nvidia's model.

With Nvidia model the track 1 can be finished without data augmentation, so it is chosen. In order to pass track 2, dropout layers and data augmentation are added.

| Layer             | Output shape | Activation | Parameters                                |
|-------------------|--------------|------------|-------------------------------------------|
|**Input**          | 66x200x3     |            | YUV color space                           |
|**Normalization**  | 66x200x3     |            | Range:(-1 , 1)                            |
|**Convolution**    | 31x98x24     | ELU        | Strides:(2, 2), Padding:valid, Kernel:5x5 |
|**Convolution**    | 14x47x36     | ELU        | Strides:(2, 2), Padding:valid, Kernel:5x5 |
|**Convolution**    |  5x22x48     | ELU        | Strides:(2, 2), Padding:valid, Kernel:5x5 |
|**Convolution**    |  3x20x64     | ELU        | Strides:(1, 1), Padding:valid, Kernel:3x3 |
|**Convolution**    |  1x18x64     | ELU        | Strides:(1, 1), Padding:valid, Kernel:3x3 |
|**Flatten**        |  1152        |            |                                           |
|**Droupout**       |  1152        |            | 0.5                                       |
|**Fully connected**|  1164        | ELU        |                                           |
|**Droupout**       |  1152        |            | 0.5                                       |
|**Fully connected**|   100        | ELU        |                                           |
|**Fully connected**|    50        | ELU        |                                           |
|**Fully connected**|    10        | ELU        |                                           |
|**Output**         |     1        |            |                                           |

Total params: 1,595,511


## Data Preparation & Training Approach
---

Before training, the data should be collected and processed into a format that can be consumed by the model.

1. Collecting data from simulator
Collecting data from simulator is a hard work. It requires running on the track in both direction, data for recovery, and to emphasize the turns, recording more clips on the turns is necesary. However, the hardest part is the controller. It can't be done without a joystick, so I ended up using data provided by Udacity.

2. Splitting dataset
Validation dataset is used to select best training result from all epochs. 10% or 20% of collected data is split into validation dataset after shuffled.

3. Generate training data and validation data
There are 2 ways of feeding data to fit the model. One is feeding in all prepared data which requires big enough memory to hold all image data. The other is using generator that loads images on the fly. Both training data and validtion data need to be processed before feeding into model. Preprocess and data augmentation will be applied which will be mentioned in the following section. Data augmentation includes randomization. Randomization process is applied each time we load images by generator. This increases the variation of the training data and validation data. This process is expected to avoid overfitting. However, it seems introduce too much variability and drag down the learning performance, so generator is not used for the end result.

4. Avoiding zero angle dominance
As seen in the follwoing diagram, the zero steering angle is dominating the dataset. In order to balance the data, a small angle dropping parameter is given. In the fixed size dataset approach, 20% of small angles will be dropped. In generator approach, the dropping rate goes from 80% decreasing half of the proportion each time down to 0. The idea came from [Scott Penberthy](https://github.com/drscott173/sdc-ghostrider). The result seems pretty good by focusing on the turns at the beginning of the training, then adding the small angles back at the later training epochs.

![](images/angle_distribution.png?raw=true "Angle distribution")

5. Early stopping
It is very hard to decide how many epochs should be used while training. Sometimes I found the validation loss drop very slowly, I would set a higher value for the minimum gap between the descreasing validation losses.


## Data Augmentation & Preprocessing
---

Preprocessing is applied to all images that is consumed by the model including the image for predictions.

1. **Cropping:** Remove unrelated parts of the image (e.g. background, car hood)

![](images/crop.png?raw=true "")

2. **Scaling:** To Improve the performance by shrinking the size

![](images/scale.png?raw=true "")

3. **Color space:** Use YUV as Nvidia model suggests.

![](images/color.png?raw=true "")

Data augmentation is very important in this project. The data provided by Udacity is not enough for training a generalized model, so we have to create some variations from the existing data to teach the model what should be ignored and what should be focused. Most of the ideas are learned from [Vivek Yadav's article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.297rvhfqn).

1. **Side cameras:** Side camera images can be used to simulate a car wandering of to the side of the road, so we can apply a small steering angle to the opposite side to recover.

![](images/side.png?raw=true "")

2. **Shift:** Shifting the image horizontally can be seen as a car on a different position on the road, so we can also apply a small steering angle to steer the car back to the center. Shifting vertically can be seen as uphill or downhill, which just make the model recognize more variations.

![](images/shift.png?raw=true "")

3. **Horizontal flip:** Flipping image can balance the number of left turns and right turns.

![](images/flip.png?raw=true "")

4. **Brightness:** Adjusting brightness can simulate the day or night conditions.

![](images/brightness.png?raw=true "")

5. **Shadow:** Adding random line crossing the image and darkening the color on the one side to simulate the shadow cast from the environment. This is a brillient idea.

![](images/shadow.png?raw=true "")

## Reflection
---
1. Randomization in splitting dataset and augmenting data cause the training result qutie unstable. Sometimes, the validation loss sticks in high number. The same setting should be trained several times to make sure it is working or not. It is probably better to generate the data and store it rather then generating it each time.

2. Data augmentation is as important as selecting model. Imbalanced data, incorrect tweaking angles, and lack of variation lead the model to bad result.

3. Accuracies in this project is useless. I spent quite a bit of time before I realized that predicted angles will almost always a little bit different from the expected angles because they are floating numbers. In this case, we only need mean squared error or mean absolute error.


## Reference
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ms9dp9y1r
https://medium.com/@acflippo/cloning-driving-behavior-by-augmenting-steering-angles-5faf7ea8a125#.9qo2nhv3o
https://github.com/drscott173/sdc-ghostrider

