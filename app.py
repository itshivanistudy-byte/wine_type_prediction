# -*- coding: utf-8 -*-

import numpy as np
import joblib
import streamlit as st

# Load model
model = joblib.load("wine_type_prediction.pkl")

# Page config
st.set_page_config(page_title="Wine Type Prediction", layout="centered")

st.title("Wine Type Classification App")
st.write("Predict whether the wine is **Red** or **White** using chemical properties")

# Inputs
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)
quality = st.number_input("Quality", min_value=0.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[ 
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        quality
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("White Wine")
    else:
        st.error("Red Wine")

