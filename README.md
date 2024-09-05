# Machine Learning Classification Project

## Overview

This project focuses on developing and evaluating various machine learning models for predicting heart disease. We explore and compare the performance of Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest Classifier. The project also includes hyperparameter tuning to optimize model performance and an analysis of feature importance.

## Table of Contents

1. [Setup](#setup)
2. [Data](#data)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Feature Importance](#feature-importance)
7. [Conclusion](#conclusion)

## Setup

To get started with this project, you'll need to have Python and the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

These libraries can be installed using Python's package manager, `pip`.

## Data

The dataset used in this project is a heart disease classification dataset. The goal is to predict the presence of heart disease based on the following features:

- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male, 0 = female)
- **cp**: Chest pain type (1-4)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol level (in mg/dl)
- **fbs**: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: Depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (1-3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
- **target**: Presence or absence of heart disease (1 = presence, 0 = absence)

## Model Training

Three different models were trained and evaluated:

1. **Logistic Regression**: This model was used to establish a baseline for performance.
2. **K-Nearest Neighbors (KNN)**: This model was evaluated for its performance and compared against the baseline.
3. **Random Forest Classifier**: This model was tested to assess its effectiveness in predicting heart disease.

## Model Evaluation

Each model's performance was assessed using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The ratio of true positives to the sum of true and false positives.
- **Recall**: The ratio of true positives to the sum of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC AUC Score**: A metric that evaluates the model's ability to distinguish between classes.

Evaluation also included the visualization of confusion matrices and classification reports.

## Hyperparameter Tuning

Hyperparameter tuning was conducted to enhance model performance. Specifically:

- **K-Nearest Neighbors (KNN)**: Various numbers of neighbors were tested to find the optimal value.
- **Logistic Regression**: GridSearchCV was employed to identify the best hyperparameters, such as the regularization parameter.

## Feature Importance

The importance of each feature was analyzed using the coefficients from the Logistic Regression model. This analysis helps in understanding the contribution of each feature to the model's predictions regarding heart disease.

## Conclusion

The Logistic Regression model with tuned hyperparameters achieved the best performance metrics. The project demonstrated the importance of hyperparameter tuning and feature importance analysis in improving model accuracy and interpretability for predicting heart disease.

For additional details, refer to the code and visualizations provided in the repository.

## Acknowledgements

- Scikit-learn for machine learning algorithms and utilities.
- Seaborn for data visualization.
- Matplotlib for plotting.
