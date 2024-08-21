# Deep-Learning-for-Brain-Cancer-Detection

## Overview

This project demonstrates the development of a deep learning model for the detection of brain tumors from MRI images. The model was built using the Keras library in R and achieved an accuracy of 96.66% on the test set.

## Dataset

The dataset used in this project was obtained from Kaggle and contains MRI images of brains with and without tumors. The images were split into training, validation, and test sets.

## Model Architecture

The final model architecture consists of the following layers:

- 4 Convolutional layers with ReLU activation and increasing filter sizes (32, 64, 128, 256)
- Max Pooling layers after each Convolutional layer
- Dropout layers for regularization
- A Flatten layer
- A Dense layer with 256 units and ReLU activation
- A final Dense layer with 1 unit and Sigmoid activation for binary classification

## Training and Evaluation

The model was trained using the Adam optimizer and binary cross-entropy loss. Data augmentation techniques, such as horizontal flipping, rotation, and zoom, were applied to the training data to improve the model's generalization. The model was evaluated on the test set, achieving an accuracy of 96.66%.

## Usage

To use the model, you can load the pre-trained weights and apply the model to new MRI images. The code for loading the model and making predictions is provided in the accompanying Jupyter Notebook.

## Conclusion

This project demonstrates the potential of deep learning techniques in the medical domain, specifically for the early detection of brain tumors. The trained model can be used as a tool to assist medical professionals in making more accurate and timely diagnoses.
