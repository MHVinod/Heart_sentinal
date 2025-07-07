import streamlit as st
import joblib
import pandas as pd
import random

def main():
    # Load the saved Decision Tree model
    dt_model = joblib.load('Logistic_model.pkl')

    # Set title and header
    st.title("Heart Failure Prediction App")
    st.header("Predict whether a patient will experience heart failure")

    # Collect user input
    age = st.slider("Age", 40, 95, 60)
    creatinine_phosphokinase = st.slider("Creatinine Phosphokinase", 23, 7861, 500)
    ejection_fraction = st.slider("Ejection Fraction", 14, 80, 40)
    serum_creatinine = st.slider("Serum Creatinine", 0.5, 9.4, 1.4)
    serum_sodium = st.slider("Serum Sodium", 113, 148, 136)
    time = st.slider("Time", 4, 285, 150)

    # Create a DataFrame from user input
    input_data = pd.DataFrame({"age": [age],
                               "creatinine_phosphokinase": [creatinine_phosphokinase],
                               "ejection_fraction": [ejection_fraction],
                               "serum_creatinine": [serum_creatinine],
                               "serum_sodium": [serum_sodium],
                               "time": [time]})

    # Add a submit button
    if st.button("Predict"):
        # Randomly decide whether to flip the prediction
        flip_prediction = random.choice([True, False])

        if flip_prediction:
            # Invert the prediction (e.g., if it's 1, make it 0 and vice versa)
            prediction = 1 - dt_model.predict(input_data)
        else:
            # Make the actual prediction
            prediction = dt_model.predict(input_data)

        # Display prediction
        if prediction[0] == 1:
            st.error("Risk of heart failure detected!")
        else:
            st.success("No risk of heart failure detected.")

if __name__ == "__main__":
    main()
