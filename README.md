# FOB_LR
Independent Study Project: Classification of Fall Out Boy Eras.
This version of the repository is public and does not contain any sensitive personal access keys.

## Abstract

The statistical modeling technique of logistic regression (LR) has become the standard for binary classification. Modern supervised machine learning often leverages LR to predict whether a given event will fall into one of two categories: Class 0 or Class 1.

Generally, this process helps data scientists learn which variables are good predictors of class membership. One business application might be loan classification—a million-dollar industry. When applied to medicine, logistic regression can predict who is more susceptible to disease; in this way, machine learning saves lives. 

For this project, I use data from Spotify and Genius to build a database consisting of songs and lyrics written by my favorite band, Fall Out Boy. I then construct a logistic regression model to classify the songs and lyrics into one of two Fall Out Boy eras: pre-hiatus or post-hiatus. Most Fall Out Boy listeners can instantly tell which era a song belongs to; this study will determine if a computer can also differentiate between the two. Additionally, I will test the regression against other binary classification algorithms, including random forest and support vector machines.

## Notebooks
- **FOB_Extract**: Music and lyric data pull using the Spotipy and LyricsGenius APIs
- **FOB_EDA**: Exploratory data analysis of the data extracted in **FOB_Extract**
- **Feature_Engineering**: Preparing data analyzed in **FOB_EDA** for ML
- **Model_Comparison**: Modeling FOB data with multiple ML algorithms and comparing results

## Models (Scripts)
- **LR.py**: Logistic Regression implemented from scratch in Python and NumPy
- **NN.py**: Logistic Regression implemented as a single-layer neural network in PyTorch