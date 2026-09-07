#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
log_model = pickle.load(open("logistic_model.pkl", "rb"))
std_sca = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction using Logistic Regression")

def user_input_parameters():

    Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, step=1)

    Glucose = st.sidebar.number_input("Glucose", min_value=0.0)

    BloodPressure = st.sidebar.number_input("Blood Pressure", min_value=0.0)

    SkinThickness = st.sidebar.number_input("Skin Thickness", min_value=0.0)

    Insulin = st.sidebar.number_input("Insulin", min_value=0.0)

    BMI = st.sidebar.number_input("BMI", min_value=0.0)

    DiabetesPedigreeFunction = st.sidebar.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        format="%.3f"
    )

    Age = st.sidebar.slider("Age", 1, 100)

    data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }

    features = pd.DataFrame(data, index=[0])

    # Scale the input
    features = pd.DataFrame(
        std_sca.transform(features),
        columns=features.columns
    )

    return features


df = user_input_parameters()

pred = log_model.predict(df)
pred_prob = log_model.predict_proba(df)

button = st.button("Predict")

if button:

    st.subheader("Prediction")

    if pred[0] == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is not likely to have Diabetes.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Diabetes: {pred_prob[0][1]:.2%}")


# In[ ]:




