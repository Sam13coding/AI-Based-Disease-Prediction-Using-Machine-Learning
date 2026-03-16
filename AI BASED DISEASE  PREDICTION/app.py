import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Based Disease Prediction",
    page_icon="🩺",
    layout="centered"
)


# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()


# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None


# ---------------- AUTH FUNCTIONS ----------------
def signup():
    st.subheader("📝 Create Account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Signup"):
        if username and password:
            try:
                c.execute(
                    "INSERT INTO users (username, password) VALUES (?,?)",
                    (username, password)
                )
                conn.commit()
                st.success("Account created successfully ✅")
            except:
                st.error("Username already exists ❌")
        else:
            st.warning("Please fill all fields")


def login():
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        result = c.fetchone()

        if result:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


# ---------------- AUTH UI ----------------
if not st.session_state.logged_in:
    st.title("🩺 AI Based Disease Prediction System using ML")

    auth_choice = st.radio("Choose Option", ["Login", "Signup"])

    if auth_choice == "Login":
        login()
    else:
        signup()

    st.stop()


# ====================== MAIN APP ======================

# ---------------- BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_model(model_name):
    with open(os.path.join(MODEL_DIR, model_name), "rb") as f:
        return pickle.load(f)


diabetes_model = load_model("diabetes_model.pkl")
heart_model = load_model("heart_model.pkl")
liver_model = load_model("liver_model.pkl")


# ---------------- SIDEBAR ----------------
st.sidebar.success(f"Logged in as {st.session_state.user}")
if st.sidebar.button("🚪 Logout"):
    logout()


# ---------------- UI TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;color:#2563eb;'>🩺 AI Based Disease Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning Powered Health Risk Detection</p>",
    unsafe_allow_html=True
)
st.divider()


# ---------------- DISEASE SELECTOR ----------------
disease = st.selectbox(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Liver Disease"]
)


# ========================= DIABETES =========================
if disease == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    blood_pressure = st.number_input("Blood Pressure", 0, 200)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
     data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])
    
     result = diabetes_model.predict(data)[0]

     if result == 1:
         st.error("⚠️ High Risk of Diabetes")
     else:
         st.success("✅ No Diabetes Detected")
    
     diabetes_df = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DPF": [dpf],
            "Age": [age]
        })


     st.subheader("📊 Patient Diabetes Parameters")
     st.bar_chart({
            "Glucose": glucose,
            "BMI": bmi,
            "Age": age,
            "Insulin": insulin
        })

    #  st.subheader("📈 Diabetes Dataset Analysis")
    #  fig, ax = plt.subplots()
    #  sns.boxplot(x="Age", y="Glucose", data=diabetes_df, ax=ax)
    #  st.pyplot(fig)
    
         
# ========================= HEART =========================
elif disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol Level", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.number_input("Resting ECG Result (0–2)", 0, 2)
    thalch = st.number_input("Maximum Heart Rate Achieved", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0)

    # Encoding categorical values
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):
        data = np.array([[age, sex, cp, trestbps, chol,
                          fbs, restecg, thalch,
                          exang, oldpeak]])

        result = heart_model.predict(data)[0]

        if result == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ No Heart Disease Detected")
          
        heart_df = pd.DataFrame({
            "age": [age],
            "chol": [chol],
            "thalch": [thalch],
            "oldpeak": [oldpeak]
        })


        st.subheader("📊 Patient Health Analysis")

        patient_data = {
            "Age": age,
            "Cholesterol": chol,
            "BP": trestbps,
            "Max HR": thalch,
            "ST Depression": oldpeak
        }

        st.bar_chart(patient_data)

        # # Dataset-level analysis
        # st.subheader("📈 Heart Disease Dataset Insights")

        # fig, ax = plt.subplots()
        # sns.histplot(data=heart_df, x="age", y="trestbps", bins=20, ax=ax)
        # st.pyplot(fig)

# ========================= LIVER =========================
elif disease == "Liver Disease":
    st.subheader("🧪 Liver Disease Prediction")

    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tot_bilirubin = st.number_input("Total Bilirubin", 0.0, 100.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 50.0)
    tot_proteins = st.number_input("Total Proteins", 0.0, 10.0)
    albumin = st.number_input("Albumin", 0.0, 6.0)
    ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0)
    sgpt = st.number_input("SGPT", 0, 300)
    sgot = st.number_input("SGOT", 0, 300)
    alkphos = st.number_input("Alkaline Phosphatase", 0, 500)

    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease"):
     data = np.array([[age, gender, tot_bilirubin, direct_bilirubin,
                      tot_proteins, albumin, ag_ratio,
                      sgpt, sgot, alkphos]])

     result = liver_model.predict(data)[0]

     if result == 1:
         st.error("⚠️ High Risk of Liver Disease")
     else:
         st.success("✅ No Liver Disease Detected")

     liver_df = pd.DataFrame({
            "Total_Bilirubin": [tot_bilirubin],
            "Albumin": [albumin],
            "SGPT": [sgpt],
            "SGOT": [sgot]
        }) 
     st.subheader("📊 Liver Health Analysis")

    liver_features = {
        "Total Bilirubin": tot_bilirubin,
        "Direct Bilirubin": direct_bilirubin,
        "Total Proteins": tot_proteins,
        "Albumin": albumin,
        "A/G Ratio": ag_ratio,
        "SGPT": sgpt,
        "SGOT": sgot,
        "Alkaline Phosphatase": alkphos
    }

    fig, ax = plt.subplots()
    ax.bar(liver_features.keys(), liver_features.values())
    ax.set_ylabel("Values")
    ax.set_title("Patient Liver Parameters")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:13px;'>© 2026 AI Based Disease Prediction | Mini Project</p>",
    unsafe_allow_html=True
)
