import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.title("Late Delivery Risk Prediction")
st.write("Predict if an order will be delivered late")

# Inputs
shipping_mode = st.selectbox(
    "Shipping Mode",
    encoder.classes_
)

real_days = st.number_input("Actual Shipping Days", min_value=0)

scheduled_days = st.number_input("Scheduled Shipping Days", min_value=0)

quantity = st.number_input("Order Item Quantity", min_value=1)

price = st.number_input("Product Price", min_value=1.0)

# Encode shipping mode
shipping_mode_encoded = encoder.transform([shipping_mode])[0]

# Prediction
if st.button("Predict Delivery Status"):

    features = np.array([[
        real_days,
        scheduled_days,
        shipping_mode_encoded,
        quantity,
        price
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Late Delivery")
    else:
        st.success("✅ Delivery Expected On Time")

