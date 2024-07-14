# UNet Segmentation Project

## Project Description
This project implements a UNet model for ship detection based on dataset https://www.kaggle.com/competitions/airbus-ship-detection/data.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [File Descriptions](#file-descriptions)
  - [data.py](#datapy)
  - [unet_model.py](#unet_modelpy)
  - [model_training.py](#model_trainingpy)
  - [model_inference.py](#model_inferencepy)
  - [utils.py](#utilspy)
  - [config](#config)
- [Usage](#usage)

## Setup Instructions
Edit the config.py file to set up your global parameters like paths to files, images sizes, model hyperparameters for training.

## File Descriptions
## data.py
This file contains functions and classes for data preparation, including loading, preprocessing the dataset for training and validation.

## unet_model.py
This file defines the UNet model architecture. The model is built using TensorFlow/Keras and includes functions for creating and compiling the model (like dice_coefficient, dice_loss)

## model_training.py
This file contains the training pipeline for the UNet model. It includes code for compiling the model, and training it using the prepared data.

## model_inference.py
This file contains code for performing inference with the trained UNet model. It includes functions for loading the model, making predictions, and visualizing the results.

## utils.py
This file contains various utility functions used throughout the project. These include helper functions for metrics calculation, image processing, and other miscellaneous tasks.

##   config.py
This file contains global parameters and configuration settings used throughout the project. It includes settings for paths, hyperparameters, and other constants.


## Usage
Training the Model
To train the UNet model, run the model_training.py script: python model_training.py

Performing Inference
To perform inference with the trained model, run the model_inference.py script: python model_inference.py

