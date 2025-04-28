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

# --- Model Training  ---
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

# --- BMI Calculation ---
bmi = weight / ((height / 100) ** 2)

# --- Determine category and color based on BMI ---
if bmi < 18.5:
    category = "Underweight"
    color = "blue"
elif 18.5 <= bmi <= 24.9:
    category = "Healthy Weight"
    color = "green"
elif 25 <= bmi <= 29.9:
    category = "Overweight"
    color = "purple"
elif 30 <= bmi <= 34.9:
    category = "Obesity Class I"
    color = "orange"
elif 35 <= bmi <= 39.9:
    category = "Obesity Class II"
    color = "darkorange"
else:
    category = "Obesity Class III"
    color = "red"

# --- Display BMI and category with color ---
# --- Display BMI below Height and Weight in the sidebar ---
st.sidebar.markdown(
    f"**Your BMI is:** <span style='color:{color}'>{bmi:.2f} ({category})</span>",
    unsafe_allow_html=True
)

diabetes = st.sidebar.radio("Do you have diabetes?", ("Yes", "No"))
blood_pressure = st.sidebar.radio("Do you have blood pressure problems?", ("Yes", "No"))
transplant = st.sidebar.radio("Have you had any transplants?", ("Yes", "No"))
chronic = st.sidebar.radio("Do you have any chronic diseases?", ("Yes", "No"))
allergies = st.sidebar.radio("Do you have any known allergies?", ("Yes", "No"))
cancer_history = st.sidebar.radio("Any history of cancer in your family?", ("Yes", "No"))
surgeries = st.sidebar.number_input("Number of major surgeries you've had", 0, 20, 0)

# --- Prepare input for prediction ---
input_data = pd.DataFrame({
    'Age': [age],
    'Weight': [weight],
    'Height': [height],
    'BMI': [bmi],
    'Diabetes': [1 if diabetes == "Yes" else 0],
    'BloodPressureProblems': [1 if blood_pressure == "Yes" else 0],
    'AnyTransplants': [1 if transplant == "Yes" else 0],
    'AnyChronicDiseases': [1 if chronic == "Yes" else 0],
    'KnownAllergies': [1 if allergies == "Yes" else 0],
    'HistoryOfCancerInFamily': [1 if cancer_history == "Yes" else 0],
    'NumberOfMajorSurgeries': [surgeries]
    # Add other binary inputs if necessary
})



# Align input features
input_data = input_data.reindex(columns=model_features, fill_value=0)

# --- Prediction ---
if st.button("Predict Premium"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Insurance Premium: ${prediction[0]:,.2f}")
