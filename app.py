import streamlit as st
import pickle
import numpy as np
# 🔥 UI DESIGN (paste here)
st.markdown("""
<style>
.stApp {
    background-color: #0B0F14;
}

h1 {
    text-align: center;
    color: #4CAF50;
    font-size: 40px;
}

.block-container {
    max-width: 700px;
    margin: auto;
}

div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    padding: 10px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.title("Breast Cancer Detection App")

st.write("Enter values:")

# Example important features (not all 30, just main ones)
radius_mean = st.number_input("Radius Mean", 0.0)
texture_mean = st.number_input("Texture Mean", 0.0)
perimeter_mean = st.number_input("Perimeter Mean", 0.0)
area_mean = st.number_input("Area Mean", 0.0)
smoothness_mean = st.number_input("Smoothness Mean", 0.0)

# Fill remaining with 0 (to make total 30 features)
features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]

# Add remaining dummy values (0)
while len(features) < 30:
    features.append(0)

# Predict button
if st.button("Predict"):
    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Malignant (Cancer detected)")
    else:
        st.success("Benign (No Cancer)")
        
