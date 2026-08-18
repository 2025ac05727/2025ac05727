# Machine Learning Assignment 2

## 1\. Problem Statement

This project implements and evaluates multiple machine learning
classification models on the Breast Cancer Wisconsin dataset.
The objective is to compare the performance of different
classification algorithms using multiple evaluation metrics.

## 2\. Dataset Description

### Dataset Name

Breast Cancer Wisconsin Dataset

### Dataset Source

The dataset is provided through scikit-learn's breast cancer dataset.

### Number of Instances

569

### Number of Features

30

### Problem Type

Binary Classification

### Target Classes

The dataset contains two target classes.

Target classes:

\['malignant' 'benign']

## 3\. GitHub Repository Link

https://github.com/2025ac05727/2025ac05727/tree/main

## 4\. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Evaluation Metrics

The following metrics were calculated:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison









|ML Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.9825|0.9954|0.9861|0.9861|0.9861|0.9623|
|kNN|0.9561|0.9788|0.9589|0.9722|0.9655|0.9054|
|Random Forest	|0.9561|0.9937|0.9589|0.9722|0.9655|0.9054|
|Naive Bayes|0.9386|0.9878|0.9452|0.9583|0.9517|0.8676|
|Decision Tree|0.9123|0.9157|0.9559|0.9028|0.9286|0.8174|

### Model Observations

#### Logistic Regression

Observation to be added based on experimental results.

#### Decision Tree

Observation to be added based on experimental results.

#### kNN

Observation to be added based on experimental results.

#### Naive Bayes

Observation to be added based on experimental results.

#### Random Forest

Observation to be added based on experimental results.

### Overall Winner

The overall winner will be identified based on the experimental evaluation results.

## 5\. Streamlit Application



Streamlit Live App: https://2025ac05727-ml-assignment.streamlit.app/



The application provides:

* Test CSV upload
* Model selection
* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* MCC
* Classification report
* Confusion matrix
* Model comparison

Live Streamlit link to be added after deployment.

## 6\. Project Structure

2025ac05727/

* app.py
* requirements.txt
* README.md
* test\_data.csv
* model\_results.csv
* model/

  * scaler.pkl
  * logistic\_regression.pkl
  * decision\_tree.pkl
  * knn.pkl
  * naive\_bayes.pkl
  * random\_forest.pkl

