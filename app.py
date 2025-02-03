import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model and scaler
st.write("Loading the model...")
try:
    model, scaler = joblib.load("random_forest_model.pkl")
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Insurance Premium Estimator")

# User inputs
age = st.number_input("Age", min_value=18, max_value=66, step=1)
weight = st.number_input("Weight (kg)", min_value=51, max_value=132)
height = st.number_input("Height (cm)", min_value=145, max_value=188)
num_surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=3, step=1)

diabetes = st.selectbox("Do you have Diabetes?", ["No", "Yes"])
blood_pressure = st.selectbox("Do you have Blood Pressure Problems?", ["No", "Yes"])
transplant = st.selectbox("Have you had a Transplant?", ["No", "Yes"])
chronic_diseases = st.selectbox("Do you have Chronic Diseases?", ["No", "Yes"])
allergies = st.selectbox("Do you have Allergies?", ["No", "Yes"])
cancer_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])

# Convert categorical inputs to 0/1 using Label Encoding
label_encoder = LabelEncoder()

# Encode the binary categorical features (Yes/No)
diabetes = label_encoder.fit_transform([diabetes])[0]
blood_pressure = label_encoder.fit_transform([blood_pressure])[0]
transplant = label_encoder.fit_transform([transplant])[0]
chronic_diseases = label_encoder.fit_transform([chronic_diseases])[0]
allergies = label_encoder.fit_transform([allergies])[0]
cancer_history = label_encoder.fit_transform([cancer_history])[0]

# Label Encode 'NumberOfMajorSurgeries'
num_surgeries = label_encoder.fit_transform([num_surgeries])[0]

# Create IBM feature from Weight and Height
ibm = np.round(weight / (height / 100) ** 2)
ibm_log = np.log(ibm)  # Log transformation

# Create input array for categorical and numeric data
categorical_data = np.array([[diabetes, blood_pressure, transplant, chronic_diseases, allergies, cancer_history]])
numeric_data = np.array([[age, ibm_log, num_surgeries]])

# Create a DataFrame for categorical data to apply Label Encoding (on the fly)
categorical_df = pd.DataFrame(categorical_data, columns=["Diabetes", "BloodPressureProblems", "AnyTransplants", "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily"])
categorical_encoded = pd.get_dummies(categorical_df, drop_first=False)

# **Important:** Use the same column names that were generated in `train.py` after encoding
categorical_encoded = categorical_encoded[['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 
                                           'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']]

# Combine the numeric features (age, ibm_log, num_surgeries) with encoded categorical features
full_input_data = np.hstack((categorical_encoded.values, numeric_data))



# Apply the same standardization to the full feature set (categorical + numeric)
full_input_scaled = scaler.transform(full_input_data)



# Make prediction using the Random Forest model
if st.button("Estimate Premium"):
    premium_price = model.predict(full_input_scaled)
    st.write(f"Estimated Premium: ₹{premium_price[0]:,.2f}")
