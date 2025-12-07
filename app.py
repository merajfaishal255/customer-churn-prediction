import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("📉 Customer Churn Prediction App")

# User Inputs
tenure = st.number_input("Tenure (months)", min_value=1, max_value=60, value=12)
charges = st.number_input("Monthly Charges", min_value=100, max_value=2000, value=500)
calls = st.number_input("Support Calls", min_value=0, max_value=20, value=1)

if st.button("Predict Churn"):
    input_data = np.array([[tenure, charges, calls]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✔ Customer will STAY")
