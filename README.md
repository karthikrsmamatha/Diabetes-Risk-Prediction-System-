# 🩺 Diabetes Risk Prediction System

## Project Overview

This project uses machine learning to predict the risk of diabetes based on patient health-related features. It covers data preprocessing, exploratory data analysis, model building, evaluation, and deployment using Streamlit.

## Business Objective

To develop a machine learning system that predicts whether a person is at risk of diabetes based on health-related factors.

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains health-related information such as:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## Project Workflow

1. Data preprocessing and exploratory data analysis
2. Outlier detection and capping
3. Feature scaling using StandardScaler
4. Logistic Regression model development
5. Model evaluation using Accuracy and ROC-AUC
6. Deployment using Streamlit

## Machine Learning Model

**Logistic Regression** was used to predict whether a patient is likely to have diabetes based on the provided health-related features.

## Model Performance

* **Accuracy:** 73.38%
* **ROC-AUC:** 0.78

## Project Files

* `Diabetes_Risk_Prediction.ipynb` – Data preprocessing, EDA, feature scaling, model building, and evaluation
* `Model_Diabetes.py` – Streamlit application for diabetes risk prediction
* `logistic_model.pkl` – Trained Logistic Regression model used for prediction
* `Scaler.pkl` – Trained StandardScaler used to preprocess user inputs
* `README.md` – Project documentation

## Deployment

The trained Logistic Regression model is deployed using **Streamlit**. Users can enter patient health information through the application and receive a diabetes risk prediction.

## Results

The Logistic Regression model achieved **73.38% accuracy** and a **0.78 ROC-AUC score**. The trained model and scaler were saved as pickle files and integrated into a Streamlit application for diabetes risk prediction.
