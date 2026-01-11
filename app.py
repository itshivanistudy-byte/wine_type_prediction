# -*- coding: utf-8 -*-
"""wine_deployment.ipynb"""

#import libraries
import numpy as np
import joblib
import streamlit as st

#wine_type_prediction.pkl
model = joblib.load('wine_type_prediction.pkl')

st.set_page_config(page_title='Wine Type Prediction', layout="centered")
st.title("Wine Type Classification App")
st.write("predict whether the wine is **red** or **white** using chemical properties")

fixed_acidity = st.number_input("value of fixed acidity")
volatile_acidity = st.number_input("value of volatile acidity")
citric_acid = st.number_input("value of citric acid")
residual_sugar = st.number_input("value of residual sugar")
chlorides = st.number_input("value of chlorides")
free_sulfur_dioxide = st.number_input("value of free sulfur dioxide")
total_sulfur_dioxide = st.number_input("value of total sulfur dioxide")
density = st.number_input("value of density")
pH = st.number_input("value of pH")
sulphates = st.number_input("value of sulphates")
alcohol = st.number_input("value of alcohol")
quality = st.number_input("value of quality")

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
        st.success("**white wine**")
    else:
        st.error("**red wine**")
