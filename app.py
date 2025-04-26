# app.py

import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Medical Premium Estimator", layout="centered")

# --- Load Dataset ---
@st.cache_data
def load_data():
    # Raw GitHub link for CSV (note: must be raw)
    url = 'https://raw.githubusercontent.com/Shantal98/Medical-Premium-Prediction/main/Medicalpremium.csv'
    df = pd.read_csv(url)
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    return df

df = load_data()

# --- Preprocessing ---
X = df.drop(columns=['PremiumPrice'])
y = df['PremiumPrice']

# Use BMI instead of Height and Weight
X['BMI'] = df['BMI']
X = X.drop(columns=['Height', 'Weight'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training (only first time or on update) ---
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

model, model_features = train_model()

# --- Streamlit UI ---

st.title("🩺 Medical Premium Estimator 🏥")
st.sidebar.header("Enter Your Details")

# --- Inputs ---
age = st.sidebar.slider("Age", 18, 100, 30)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 250, 170)
diabetes = st.sidebar.radio("Do you have diabetes?", ("Yes", "No"))
blood_pressure = st.sidebar.radio("Do you have blood pressure problems?", ("Yes", "No"))

# --- Prepare input for prediction ---
input_data = pd.DataFrame({
    'Age': [age],
    'Weight': [weight],
    'Height': [height],
    'Diabetes': [1 if diabetes == "Yes" else 0],
    'BloodPressureProblems': [1 if blood_pressure == "Yes" else 0],
    # Add other binary inputs if necessary
})

input_data['BMI'] = input_data['Weight'] / ((input_data['Height'] / 100) ** 2)
input_data = input_data.drop(columns=['Weight', 'Height'])

# Align input features
input_data = input_data.reindex(columns=model_features, fill_value=0)

# --- Prediction ---
if st.button("Predict Premium"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Insurance Premium: ${prediction[0]:,.2f}")
