# Fallen Tree Image Classifier

This project uses a convolutional neural network (CNN) to classify images of forested areas, predicting whether the image contains any fallen trees. It is a binary image classification model and uses TensorFlow and Keras.

## Project Overview

-Goal: Identify whether a given image of trees contains at least one fallen tree
-Approach: Built two different CNNs, one trained from scratch using a csv to load images and one separating images into their respective folders and built off of existing models.
- Data Format: JPEG images with associated labels 
- Frameworks Used: Tensorflow, Keras, Pandas, scikit-learn


## How to run

### Install Dependencies

This requires Python 3.8 with tensor flow, pandas, scikit-learn, pandas, and numpy installed.

### Prepare the dataset

Each of the three folders (train, valid, test) inside Data/ should contain: 

- A labels.ccsv file with the columns:
Filename, fallen
Image1.jpg, 0
Image2.jpg,2

- The actual image files referenced by filename.

### Train the model

From the root folder can run

Python train_fallen_tree_classifier.py

The script will then:

- Load the images and labels

- Train the CNN with data augmentation

- Evaluate on the test set

- Print a confusion matrix and classification report

- Save the trained model to a folder named "final_fallen_tree_model"

## Model Performance

Example evaluation on test set 

Test Accuracy: 89.83%

Confusion Matrix:
[[42  2]
 [ 4 11]]

Classification Report:
              precision    recall  f1-score   support
  Not Fallen       0.91      0.95      0.93        44
      Fallen       0.85      0.73      0.79        15


## Output

- final_fallen_tree_model/ : Saved tensor flow model, can reload it with tf.keras.models.load_model("final_fallen_tree_model")

