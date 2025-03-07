import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.title("Air Quality Predicton System")

# Upload dataset
uploaded_file = st.file_uploader("air_quality_data(1).csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Handle missing values
    df = df.dropna()
    
    # Select target and features
    target_column = st.selectbox("Select target variable (e.g., AQI, PM2.5)", df.columns)
    feature_columns = st.multiselect("Select features for training", [col for col in df.columns if col != target_column])
    
    if feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        with open("air_quality_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Model Evaluation")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(feature_columns, feature_importance)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
        
        # User input for prediction
        st.subheader("Make a Prediction")
        user_inputs = []
        for feature in feature_columns:
            user_value = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
            user_inputs.append(user_value)
        
        if st.button("Predict AQI"):
            with open("air_quality_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)
            prediction = loaded_model.predict([scaler.transform([user_inputs])[0]])
            st.success(f"Predicted {target_column}: {prediction[0]:.2f}")
else:
    st.write("Waiting for file upload...")