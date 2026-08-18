# Machine Learning Assignment 2

## M.Tech (AI/ML) - Machine Learning

\---

## 1\. Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models on a common classification dataset.

The implemented models are evaluated using the following evaluation metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

The trained classification models are also integrated into an interactive Streamlit web application to demonstrate model evaluation and prediction.

\---

## 2\. Dataset Description

### Breast Cancer Wisconsin (Diagnostic) Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset is a binary classification dataset used to classify breast tumors as malignant or benign.

### Dataset Details

* Number of instances: 569
* Number of features: 30
* Classification type: Binary classification
* Target variable: Tumor diagnosis
* Classes: Malignant and Benign

The dataset satisfies the assignment requirements of a minimum of 500 instances and a minimum of 12 features.

The dataset was loaded using the Breast Cancer dataset available through the Scikit-learn library.

\---

## 3\. GitHub Repository Link

**GitHub Repository:**



https://github.com/2025ac05727/2025ac05727/tree/main



## 4\. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble Model)

All five models were trained and evaluated using the same dataset and the same train-test split.

\---

## 5\. Model Evaluation and Comparison

The models were evaluated using:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Comparison Table

|ML Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.9825|0.9954|0.9861|0.9861|0.9861|0.9623|
|Decision Tree|0.9123|0.9157|0.9559|0.9028|0.9286|0.8174|
|kNN|0.9561|0.9788|0.9589|0.9722|0.9655|0.9054|
|Naive Bayes|0.9386|0.9878|0.9452|0.9583|0.9517|0.8676|
|Random Forest (Ensemble)|0.9561|0.9937|0.9589|0.9722|0.9655|0.9054|

\---

## 6\. Model Performance Observations

### Logistic Regression

Logistic Regression achieved the best overall performance among all five models.

It achieved an Accuracy of 0.9825, AUC of 0.9954, Precision of 0.9861, Recall of 0.9861, F1 Score of 0.9861, and MCC of 0.9623.

Logistic Regression obtained the highest value for all six evaluation metrics in this experiment. Therefore, it demonstrated the strongest overall classification performance on the selected dataset.

### Decision Tree

Decision Tree achieved an Accuracy of 0.9123, AUC of 0.9157, Precision of 0.9559, Recall of 0.9028, F1 Score of 0.9286, and MCC of 0.8174.

Among the five models, Decision Tree showed the weakest overall performance based on the experimental evaluation results. Its Accuracy, AUC, F1 Score, and MCC were lower than those of the other models.

### kNN

The kNN classifier achieved an Accuracy of 0.9561, AUC of 0.9788, Precision of 0.9589, Recall of 0.9722, F1 Score of 0.9655, and MCC of 0.9054.

The model demonstrated strong classification performance. Its Recall of 0.9722 indicates that it correctly identified a high proportion of the positive class. Its F1 Score of 0.9655 also indicates a good balance between Precision and Recall.

### Naive Bayes

Naive Bayes achieved an Accuracy of 0.9386, AUC of 0.9878, Precision of 0.9452, Recall of 0.9583, F1 Score of 0.9517, and MCC of 0.8676.

Naive Bayes achieved a high AUC of 0.9878, indicating strong ability to distinguish between the two classes. However, its overall classification metrics were lower than those of Logistic Regression, kNN, and Random Forest.

### Random Forest (Ensemble)

Random Forest achieved an Accuracy of 0.9561, AUC of 0.9937, Precision of 0.9589, Recall of 0.9722, F1 Score of 0.9655, and MCC of 0.9054.

Random Forest demonstrated strong classification performance. Its AUC of 0.9937 was very close to the AUC achieved by Logistic Regression. However, its Accuracy, Precision, Recall, F1 Score, and MCC were lower than those of Logistic Regression.

### Overall Winner

Logistic Regression is the overall winner for this dataset.

It achieved the highest values for all six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

Therefore, based on the experimental evaluation, Logistic Regression provided the best overall performance among the five classification models tested.

\---

## 7\. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

The Streamlit application provides the following features:

* Test data CSV upload
* Model selection
* Evaluation metrics
* Confusion matrix
* Classification report
* Results for the selected classification model

### Live Streamlit App

&#x20;https://2025ac05727-ml-assignment.streamlit.app/

\---

## 8\. Project Structure

```text
2025ac05727/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- test\_data.csv
|
`-- model/
    |-- scaler.pkl
    |-- logistic\_regression.pkl
    |-- decision\_tree.pkl
    |-- knn.pkl
    |-- naive\_bayes.pkl
    `-- random\_forest.pkl
```

\---

## 9\. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 split.

* Training samples: 455
* Testing samples: 114

Stratified splitting was used to maintain the class distribution between the training and testing datasets.

Feature standardization was performed using StandardScaler for models where feature scaling is important, including Logistic Regression and kNN.

\---

## 10\. Conclusion

Five classification models were implemented and evaluated on the Breast Cancer Wisconsin (Diagnostic) dataset.

The experimental results show that Logistic Regression achieved the best performance across all six evaluation metrics. Random Forest and kNN also demonstrated strong performance, while Decision Tree produced the lowest overall performance among the evaluated models.

The trained models were prepared for integration into an interactive Streamlit application for model selection, evaluation, and visualization of classification results.

\---

## 11\. GitHub Repository

**Repository Link:**

https://github.com/2025ac05727/2025ac05727/tree/main



## 12\. Streamlit Application

**Live Application Link:**

&#x20;https://2025ac05727-ml-assignment.streamlit.app/

