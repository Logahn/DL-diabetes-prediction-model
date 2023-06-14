# Deep Learning Diabetes Prediction Model

This code is a Jupyter Notebook that uses several machine learning techniques to predict diabetes based on a number of features.

## Prerequisites

This code requires the following libraries to be installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

You can install the required libraries using pip:

```zsh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

To use this code, simply run the Jupyter Notebook.

## Code Overview

### Import Required Libraries

This section imports a number of libraries including pandas, numpy, and matplotlib.

### Load the Data

The code reads in a CSV file called diabetes.csv and prints out information about the data, including the shape of the data and a count of the data grouped by the Outcome column.

### K-Nearest Neighbors to Predict Diabetes

This section uses the K-Nearest Neighbors (KNN) algorithm to predict diabetes. It splits the data into train and test sets and then builds and tests several KNN models. It then plots the accuracy of each model and selects the best model to make predictions.

### Logistic Regression

This section uses logistic regression to predict diabetes. It builds several models with different values of the regularization parameter C and prints out the accuracy of each model.

### Decision Tree Classifier to Predict Diabetes

This section uses a decision tree classifier to predict diabetes. It builds several models with different maximum depths and then prints out the accuracy of each model.

### Feature Importance in Decision Trees

This section prints out the feature importances of the decision tree classifier model.

### Deep Learning to Predict Diabetes

This section uses a multi-layer perceptron (MLP) neural network to predict diabetes. It builds a model and prints out the accuracy of the model. It then uses a StandardScaler to scale the data and builds another model, printing out its accuracy. Finally, it plots the weight matrix of the first layer of the model.```

### Link

Google colab file is located at https://colab.research.google.com/drive/1X7QSNZ08-b65-TE2MLBSm8SMSRiy3e4p
