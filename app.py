import streamlit as st
import pandas as pd
import pickle
import os

# File paths
model_file = r"D:\year 3 sem 2\Nature Inspired Algorithms\mini project\knn_model.pkl"
scaler_file = r"D:\year 3 sem 2\Nature Inspired Algorithms\mini project\scaler.pkl"

# --- Load model ---
if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"❌ Model file not found at {model_file}")
    st.stop()

# --- Load scaler ---
if os.path.exists(scaler_file):
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
else:
    st.error(f"❌ Scaler file not found at {scaler_file}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("⚙️ App Settings")
st.sidebar.info(
    "Welcome to the **Diabetes Predictor \n\n"
    "Enter your health details on the right and click **Predict**.\n\n"
    "The app uses a trained **KNN model** to estimate diabetes risk."
)

# --- Main Title ---
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'> Diabetes Risk Predictor</h1>",
    unsafe_allow_html=True,
)

st.markdown("---")

# --- Input sections ---
with st.expander("👶 Demographics", expanded=True):
    Age = st.number_input("Age (years)", min_value=1, max_value=120, value=35, help="Enter your age in years")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of pregnancies (for women)")

with st.expander("🧪 Medical Measurements", expanded=True):
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120, help="Plasma glucose concentration")
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70, help="Diastolic blood pressure (mm Hg)")
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85, help="2-Hour serum insulin (mu U/ml)")
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, help="Body Mass Index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5,
        help="Function that scores likelihood of diabetes based on family history"
    )

st.markdown("---")

# --- Prediction Button ---
if st.button("Predict Diabetes Risk"):
    # Create input DataFrame
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
        columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 📊 Prediction Result:")

    # Progress bar for probability
    st.progress(int(probability * 100))

    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Diabetes** \n\n Probability: **{probability:.2f}**")
        st.markdown("<div style='background-color:#ffcccc;padding:10px;border-radius:10px;'>"
                    "🔴 Please consult a healthcare professional for further tests."
                    "</div>", unsafe_allow_html=True)
    else:
        st.success(f"✅ **Low Risk of Diabetes** \n\n Probability: **{probability:.2f}**")
        st.markdown("<div style='background-color:#ccffcc;padding:10px;border-radius:10px;'>"
                    "🟢 Keep maintaining a healthy lifestyle!"
                    "</div>", unsafe_allow_html=True)

