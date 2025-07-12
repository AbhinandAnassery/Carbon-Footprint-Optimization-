import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Green Route Optimizer", layout="wide")
st.title("🌿 Carbon Footprint Route Optimizer for Supply Chain")

# Upload F.csv
uploaded_file = st.file_uploader("Upload F.csv")

# Fix corrupted lines
def clean_corrupted_csv(file, expected_cols=28):
    raw = file.read().decode("utf-8")
    lines = raw.splitlines()
    header = lines[0]
    fixed = [header]
    for line in lines[1:]:
        line = re.sub(r"(?<=[0-9])(?=\d{2}\.)", ",", line)
        if line.count(",") == expected_cols - 1:
            fixed.append(line)
    return io.StringIO("\n".join(fixed))

if uploaded_file:
    try:
        fixed_data = clean_corrupted_csv(uploaded_file)
        df = pd.read_csv(fixed_data)

        st.success("✅ F.csv loaded and cleaned!")
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        target = 'fuel_consumption_rate'
        drop_cols = ['timestamp', 'risk_classification']

        if target not in df.columns:
            st.error(f"❌ Target column '{target}' not found.")
            st.stop()

        df[target] = pd.to_numeric(df[target], errors='coerce')
        df.dropna(subset=[target], inplace=True)

        X = df.drop(columns=[target] + drop_cols, errors='ignore')
        X = X.apply(pd.to_numeric, errors='coerce')
        X.dropna(axis=1, inplace=True)
        X.fillna(X.mean(), inplace=True)

        if X.empty:
            st.error("❌ No numeric features left after cleaning.")
            st.stop()

        y = df[target]

        # Scale and split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        with st.spinner("🚀 Training Random Forest Model..."):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)

        st.success(f"📊 MAE: {mae:.2f} | R² Score: {r2:.2f}")

        # Plot actual vs predicted
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # Prediction section
        st.subheader("🛣️ Predict Fuel Consumption for New Input")
        user_input = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            user_input.append(val)

        if st.button("🔍 Predict"):
            user_arr = scaler.transform([user_input])
            pred = model.predict(user_arr)[0]
            co2 = pred * 23.92
            st.success(f"💧 Estimated Fuel Consumption: {pred:.2f} L/100km")
            st.info(f"🌍 Estimated CO₂ Emission: {co2:.2f} g/km")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📎 Please upload your F.csv to continue.")
