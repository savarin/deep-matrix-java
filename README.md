# Predictive modeling with neural networks

## Introduction

The goal of this project is to develop a neural network-based predictive model from scratch.

We’ll start with classes and methods to represent, load, and clean data. We then split the clean data into training and test sets. We’ll train our model with the training set - this is a process where the model allocates higher weights to more predictive columns in the data. The trained model will then be used to make predictions on the test set. Finally, we’ll evaluate model performance by comparing predictions made by our model against the actual labels in the test set.

In particular, we seek to utilize multi-threading in Java to enable us to train models in parallel. The model will be designed as a single-layer neural network (i.e. weights represented by a single matrix), and the training process will involve loss minimization via gradient descent.

## Quickstart

To run the model with the sample dataset provided, enter the following commands in terminal:

```
javac Main.java
java Main
```
A custom dataset may be used, but requires changing the `fileName` string in `Main.java`.

## Step 1: Pre-processing

In predictive modeling, the dataset consists of features and labels. Features (or signals) represent attributes that describe or pertain to each data point. Labels (or targets) represent the outcome of that data point that we seek to predict.

The sample dataset is the Kaggle Titanic dataset, where each row in the dataset represents a passenger on the Titanic. The features are the passenger class, age, sex and fare of each passenger. The label is whether or not that passenger survived the journey.

### Feature scaling

The features are scaled so that each feature has mean 0 and standard deviation 1. For simplicity, the scaling process is done prior to the train-test split.


### Train-test split

The dataset is split 80-20 into a training set and a test set. The training set is used to 'train' the model, whereas the test set is used to evaluate the model's performance.

### Parallel split

The training set is further divided into two sets of comparable size. A model would be trained separately (and in parallel) on each dataset, and the final model would be the 'average' of the two models.

## Step 2: Model training

In the model training process, the model would 'learn' how each feature impacts the label. For example, if age is a stronger predictor of survivability compared to fare, age will be given a higher 'weight' and thus more importance in calculating the probability that the passenger survived.

The model weights are represented by a single matrix, acting on a column vector of features. This, under the hood, is equivalent to a single-layer fully-connected neural network model.

### Gradient descent

The learning process involves running an optimization process called gradient descent to minimize loss. Loss is defined as the logarithm of the difference between the predicted probability of the correct label and the 'actual' probability (i.e. 1).

Gradient descent changes the weights gradually in the 'direction' that decreases average loss. Suppose we were to draw a 3-dimensional surface where the x- and y-axis are features and the z-axis the loss, then gradient descent can be visualized as moving 'down' the surface along the slope of steepest descent. The size of each step is referred to as the learning rate.

### ReLU

While matrix operations are linear, there could be a non-linear relationship between the features and the label. Introducing a rectified linear unit or ReLU layer, defined as `f(x) = max(0, x)`, can help the model capture this interaction. ReLU is widely used as its simplicity allows for much faster training without a high cost to accuracy.

### Dropout

The dropout layer can be thought of as a form of sampling, where output values are randomly set to zero by a pre-specified probability. This creates a more robust network as the process prevents interdependence, and as such the model is less likely to overfit on the training data. It is surprisingly effective, which has made it an active area of research.

## Step 3: Model evaluation

To get a sense of how good our trained model is, we 'feed' features from the test set into the model to obtain an output of label predictions. The model predictions would then be compared against the actual labels in the test set.

### Accuracy

This metric measures the proportion of correct predictions made. For the sample dataset, this would involve dividing the number of correct predictions (whether the passenger survived or did not survive) by the number of predictions made.

### Precision

This metric measures the proportion of correctly identified positives over all positive predictions made. This involves dividing the number of correct survived predictions (true positives) by the number of all survived predictions made (true positives + false positives).

### Recall

This metric measures the proportion of positives that were correctly identified. This involves dividing the number of correct survived predictions (true positives) by the actual number of passengers who survived (true positives + false negatives).
