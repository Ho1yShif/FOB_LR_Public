# FOB_LR
Independent Study Project: Classification of Fall Out Boy Eras. <br>
This version of the repository is public and does not contain any sensitive personal access keys.

## [Project Report](https://docs.google.com/document/d/1r6A2JHU6jtBoxAq3eWm2eP08l_u6KRtKTEpcoQXS84I/edit?usp=sharing)

## [Journal Article](https://arestyrurj.libraries.rutgers.edu/index.php/arestyrurj/article/view/232)

## Abstract

The statistical modeling technique of logistic regression (LR) has become the standard for binary classification. Supervised machine learning often leverages LR to predict whether a given event will fall into one of two categories: Class 0 or Class 1.

Generally, this process helps data scientists learn which variables are good predictors of class membership. One business application might be the million-dollar industry of loan classification. When applied to medicine, logistic regression can predict who is more susceptible to disease; in this way, machine learning saves lives. 

For this project, I used data from Spotify and Genius to build a dataset consisting of songs and lyrics written by my favorite band, Fall Out Boy. I then constructed a logistic regression model to classify the songs and lyrics into one of two Fall Out Boy eras: pre-hiatus or post-hiatus.

Most Fall Out Boy listeners can instantly tell which era a song belongs to– This study determined whether a computer could also differentiate between the two. Additionally, I tested the regression against other binary classification algorithms, including Random Forest and Support Vector Machines.

## Notebooks
- [**FOB_Extract**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/notebooks/FOB_Extract.ipynb): Music and lyric data pull using the Spotipy and LyricsGenius APIs
- [**FOB_EDA**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/notebooks/FOB_EDA.ipynb): Exploratory data analysis of the data extracted in **FOB_Extract**
- [**Feature_Engineering**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/notebooks/Feature_Engineering.ipynb): Preparing data analyzed in **FOB_EDA** for ML
- [**Model_Comparison**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/notebooks/Model_Comparison.ipynb): Modeling FOB data with multiple ML algorithms and comparing results

## Models (Scripts)
- [**LR.py**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/models/LR.py): Logistic Regression implemented from scratch in Python and NumPy
- [**NN.py**](https://github.com/Ho1yShif/FOB_LR_Public/blob/main/models/NN.py): Logistic Regression implemented as a single-layer neural network in PyTorch
