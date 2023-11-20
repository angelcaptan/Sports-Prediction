import pickle
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained Keras model
#model = load_model('best_mlp_model.pkl')
with open('scaler.pkl', 'rb') as file:
    scaler  = pickle.load(file)

with open('best_mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the features used during training
features = ['TotalCharges',	'MonthlyCharges','tenure','Contract_Month-to-month','OnlineSecurity_No',
            'PaymentMethod_Electronic check', 'gender',	'TechSupport_No',	'PaperlessBilling',	'Partner']


def preprocess_input(input_data):
    # Create a DataFrame with the provided input data
    input_df = pd.DataFrame([input_data], columns=features)

    # Scale the features using the pre-trained scaler
    input_scaled = scaler.transform(input_df)

    return pd.DataFrame(input_scaled, columns=input_df.columns)

def predict_churn(input_data):
    # Preprocess the input data
    input_scaled = preprocess_input(input_data)

    # Make a prediction using the loaded Keras model
    prediction = model.predict(input_scaled)
    print(prediction)

    return prediction[0]

def main():
    st.title('Customer Churn Prediction')

    # Create input fields for each feature
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(feature, value=0.0)

    # Create a button to make predictions
    if st.button('Predict Churn'):
        # Make a prediction using the provided input data
        prediction = predict_churn(input_data)

        # Display the prediction result
        st.success(f'Churn Prediction: {round(prediction * 100, 2)}%')

if __name__ == '__main__':
    main()
