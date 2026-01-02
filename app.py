
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("💓 Heart Disease Risk Predictor")
st.write("Educational purpose only – not replacement for a doctor")

# ================= LOAD DATA =================
df = pd.read_csv("Heart_Disease_Prediction.csv")

# ================= TARGET COLUMN =================
y = df["Heart Disease"]           # Correct target column
X = df.drop("Heart Disease", axis=1)

# ================= YES / NO ENCODING =================
# Convert 'Yes'/'No' to 1/0
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].map({"Yes": 1, "No": 0})

# ================= MODEL =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ================= USER INPUT =================
st.subheader("Enter Health Details")

# Manually define input fields for clarity
user_data = []
user_data.append(st.slider("Age", 10, 100, 40))
user_data.append(st.selectbox("Sex (0=Female, 1=Male)", [0, 1]))
user_data.append(st.number_input("Chest pain type", 0, 3, 0))
user_data.append(st.number_input("BP", 0, 300, 120))
user_data.append(st.number_input("Cholesterol", 0, 600, 200))
user_data.append(st.selectbox("FBS over 120 (0=No, 1=Yes)", [0, 1]))
user_data.append(st.number_input("EKG results", 0, 2, 0))
user_data.append(st.number_input("Max HR", 0, 250, 150))
user_data.append(st.selectbox("Exercise angina (0=No, 1=Yes)", [0, 1]))
user_data.append(st.number_input("ST depression", 0.0, 10.0, 0.0))
user_data.append(st.number_input("Slope of ST", 0, 2, 1))
user_data.append(st.number_input("Number of vessels fluro", 0, 3, 0))
user_data.append(st.number_input("Thallium", 0, 3, 1))

# ================= PREDICTION =================
input_array = np.array(user_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

st.subheader("🩺 Result")

if prediction == 1:
    st.error(f"⚠️ HIGH RISK ({prob*100:.1f}%)")
    st.write("Advice: Walk daily, healthy diet, BP & cholesterol control.")
else:
    st.success(f"✅ LOW RISK ({prob*100:.1f}%)")
    st.write("Good lifestyle – keep it up 🌱")
