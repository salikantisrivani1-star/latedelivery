import streamlit as st
import pickle
import numpy as np

st.title("Late Delivery Risk Prediction")

try:
    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
except:
    st.error("Model files not found. Please upload model.pkl and encoder.pkl.")
    st.stop()

shipping_mode = st.selectbox(
    "Shipping Mode",
    ["Standard Class","Second Class","First Class","Same Day"]
)

actual_days = st.number_input("Actual Shipping Days",1,10)
scheduled_days = st.number_input("Scheduled Shipping Days",1,10)
order_quantity = st.number_input("Order Quantity",1,20)
product_price = st.number_input("Product Price",1.0,1000.0)

if st.button("Predict"):

    mode = encoder.transform([shipping_mode])[0]

    input_data = np.array([[mode,actual_days,scheduled_days,order_quantity,product_price]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Late Delivery")
    else:
        st.success("Delivery Likely On Time")


    if prediction[0] == 1:
        st.error("⚠ High Risk of Late Delivery")
    else:
        st.success("✅ Delivery Expected On Time")
