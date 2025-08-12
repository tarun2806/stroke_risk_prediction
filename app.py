import streamlit as st
import pandas as pd
import joblib

# Load dictionary-based model
stroke_model = joblib.load("model.joblib")

# Page config
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")

st.title("🧠 Stroke Prediction App")
st.markdown("Enter patient details below to check stroke risk.")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Government job", "Children", "Never Worked"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, step=0.1)

# Mapping for work type
work_type_mapping = {
    "Government job": "Govt_job",
    "Children": "children",
    "Never Worked": "Never_worked",
    "Private": "Private",
    "Self-employed": "Self-employed"
}

# Predict button
if st.button("🔍 Predict Stroke Risk"):
    # Build input dictionary
    single_input = {
        "gender": gender.lower(),
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married.lower(),
        "work_type": work_type_mapping.get(work_type, work_type),
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status.lower(),
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([single_input])

    # Extract from saved model dict
    encoded_cols = stroke_model["encoded_cols"]
    numeric_cols = stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]
    model = stroke_model["model"]

    # Transform categorical features
    encoded_df = pd.DataFrame(
        preprocessor.transform(input_df),
        columns=encoded_cols
    ).reset_index(drop=True)

    # Keep numeric features
    numeric_df = input_df[numeric_cols].reset_index(drop=True)

    # Combine numeric + encoded
    X = pd.concat([numeric_df, encoded_df], axis=1)

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    # Display result
    st.subheader("📊 Prediction Result")
    st.progress(int(probability * 100))

    if prediction == 1:
        st.error(f"⚠ High Risk of Stroke — Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Stroke — Probability: {probability:.2%}")
