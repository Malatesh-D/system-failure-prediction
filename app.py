import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- 1. LOAD THE SAVED MODEL AND DATA ---

# Load the trained RandomForest model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved model columns
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

# Create the SHAP explainer for the loaded model
explainer = shap.TreeExplainer(model)


# --- 2. DEFINE THE APP INTERFACE ---

st.set_page_config(page_title="Predictive Maintenance App", layout="wide")
st.title("⚙️ Predictive Maintenance Web App")
st.write("This app predicts machine failure based on real-time sensor data. Adjust the sliders and inputs below to see the model's prediction.")

st.sidebar.header("Input Sensor Data")

# Create a function to get user input from the sidebar
def user_input_features():
    air_temp = st.sidebar.slider('Air temperature [K]', 295.0, 305.0, 298.1)
    process_temp = st.sidebar.slider('Process temperature [K]', 305.0, 315.0, 308.6)
    rotational_speed = st.sidebar.slider('Rotational speed [rpm]', 1150, 2900, 1538)
    torque = st.sidebar.slider('Torque [Nm]', 3.8, 76.6, 39.5)
    tool_wear = st.sidebar.slider('Tool wear [min]', 0, 253, 108)
    machine_type = st.sidebar.selectbox('Machine Type', ('L', 'M', 'H'))

    # Create a dictionary of the input data
    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type_H': 1 if machine_type == 'H' else 0,
        'Type_L': 1 if machine_type == 'L' else 0,
        'Type_M': 1 if machine_type == 'M' else 0
    }
    
    # Convert the dictionary to a pandas DataFrame in the correct column order
    features = pd.DataFrame(data, index=[0])
    return pd.DataFrame(features, columns=model_columns)

# Get the user input
input_df = user_input_features()


# --- 3. MAKE PREDICTIONS AND DISPLAY RESULTS ---

st.subheader("Input Features")
st.write("Current sensor readings you've set:")
st.dataframe(input_df)

# Create a button to make a prediction
if st.button('Predict Failure'):
    # Make prediction
    prediction_proba = model.predict_proba(input_df)
    failure_probability = prediction_proba[0][1] # Probability of class 1 (Failure)

    st.subheader("Prediction Result")
    
    # Display the probability with a color-coded message
    if failure_probability > 0.5:
        st.error(f"High risk of failure! Probability: {failure_probability:.2%}")
    else:
        st.success(f"Low risk of failure. Probability: {failure_probability:.2%}")

    # --- 4. EXPLAIN THE PREDICTION WITH SHAP (CORRECTED VERSION) ---
    st.subheader("Prediction Explanation")

    # Calculate SHAP explanations using the modern, stable syntax
    shap_explanation = explainer(input_df)

    # Create a SHAP waterfall plot for the 'Failure' class (class 1)
    st.write("The plot below shows how each feature contributed to the final prediction.")

    # Create the plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_explanation[0, :, 1], show=False)
    st.pyplot(fig, bbox_inches='tight')
    