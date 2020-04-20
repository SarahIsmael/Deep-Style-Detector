# Handwritten-style-recognition

### Trained Models
The trained models are available here: https://drive.google.com/drive/folders/1foLYb2733MuIYYrhT76Fw4aGsvHdogXV?usp=sharing

## E-codices dataset
### DataCollection.py
In this file the dataset (images of pages from medieval manuscripts in latin language) are collected from e-codices website project using API
### DeepStyleDetector.py
This is a main file in the project. In this file, the dataset is preprocessed, model generated, trained and tested on e-codices dataset
### dataset_train_info.tsv
information  about images are available in this file such as link of pages, dating and the library that the image resources' phisically are available

## CLamm dataset
### ClammDataCollection.py
In this file CLAMM dataset  are preprocessed such as labeled and seperated into training set and test set
### DeepStyleDetectorCLAMM.py
In this file, the data is preprocessed, model generated, trainind and tested on CLamm dataset
