# Fallen Tree Image Classifier

This project uses a convolutional neural network (CNN) to classify images of forested areas, predicting whether the image contains any fallen trees. It is a binary image classification model and uses TensorFlow and Keras.

## Project Overview
The goal of this project was to try and identify whether a given image of trees contains at least one fallen tree within them. This was formed as a binary classifation problem, but has the potential to be expaned on in future iterations. The approach decided on was to compare building a model from scratch as well as building one based off of MobileNetV2. The model trained from scratch utilized a csv to label images and the pretrained model had us organize images into their specific folders. All images are saved as jpg files. Images were organized via roboflow and a baseline was created via it's built in object detection model. The model roboflow utilized managed to get an overall precision score of 66.4%, and a recall of 52.0%.

<img width="956" height="1161" alt="results (1)" src="https://github.com/user-attachments/assets/1c54fd4a-f46f-4a9a-82f5-c1fae36e8506" />

Although the transfer learning model had theoretical advantages, in the end the simpler, scratch-built CNN which was trained using labels from CSV ultimately provided the best performance in terms of both validation accuracy and classification balance. This model managed to increase our precision score from 66 to 88% and our recall from 52 to 90%.

### Dataset

The dataset this model was trained on was put together via roboflow's public datasets and can be obtained here: 
https://app.roboflow.com/workspace-1ughj/trees-zq5p9-erdgz/5/export
When downloading select the multi-label classification csv filetype. Download the zip version, which will come with the images and label file, and unzip it in the Data folder. All organization and preprocessing of this dataset has been done already. It consists of a total of 398 images with a 70/15/15 Train/Validation/Test split. 265 files do not include fallen trees, and 133 images include fallen trees. 

Images with fallen tree's labeled
<img width="585" height="462" alt="Screenshot 2025-08-07 at 12 38 02 AM" src="https://github.com/user-attachments/assets/7b952f04-5f15-4875-8f2e-48da2f857189" />



If additional images are being added insure the images are put into the proper folders and added to the respective csv. The csv format is as follows:
filename, fallen
image_name.jpg,0
image2_name.jpg,1
...

## How to run

### Install Dependencies

This requires Python 3.8 with tensor flow, pandas, scikit-learn, pandas, and numpy installed. If python 3.8 is already installed, the rest may be installed by running ```pip install pandas tensorflow scikit-learn matplotlib numpy```

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

The final model chosen had a test accuracy of %, correctly classifying all but 2 images which did not contain fallen trees, and 11 out of 15 of the images that did contain fallen trees. Considering how tricky some of the images used were though this was a great success overall and proved way more efficient than I anticipated it would. 

<img width="349" height="280" alt="Screenshot 2025-08-07 at 1 22 19 AM" src="https://github.com/user-attachments/assets/b59e1788-8613-4434-b4e4-20f0fe02ee01" />


Confusion Matrix:

[[41  3]

 [ 2 13]]

Classification Report:

|  | precision | recall | f1-score | support |
| :------ | :------ |  :------ |  :------ | :------ | 
| Not Fallen | 0.95 | 0.93 | 0.94 | 44 | 
| Fallen | 0.81 | 0.87 | 0.84 | 15 |
|  |  |  |  |  |
| accuracy |     |      | 0.92 | 59 |
|macro avg | 0.88 | 0.90 | 0.89 | 59 |
|weighted avg | 0.92 | 0.92 | 0.92 | 59 |



## Difficulties

The first issue I ran into was that the datasets skew led the model to predict every image as not containing fallen trees. This was somewhat expected at first, as an upright tree and a fallen tree both look very similar, and was even a part of why I chose to do this task as I saw it as a good challenge for the model to take on. After some additional tuning though we managed to get really meaningful results. That aside, simply making this model was a very lengthy process which required teaching myself how to actually code image classifcation models altogether as I had never truly seen nor attempted similar tasks before. So simply learning about the structure of these models and the different metrics and routes to tune a model was over a week long endeavor.

## Output

- final_fallen_tree_model/ : Saved tensor flow model, can reload it with tf.keras.models.load_model("final_fallen_tree_model")

