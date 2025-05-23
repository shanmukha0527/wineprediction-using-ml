import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load data
@st.cache_data
def load_data():
    # Load the CSV file (not Excel)
    data = pd.read_csv("winequality-red.csv",sep=";")  # You can change path if needed

    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    # Check for expected 'quality' column
    if 'quality' not in data.columns:
        st.error("'quality' column not found. Available columns: " + str(data.columns.tolist()))
        st.stop()

    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop any rows with missing values (optional)
    data.dropna(inplace=True)

    return data

# Data Preprocessing and Model Training
@st.cache_data
def train_model():
    data = load_data()

    # Feature and target separation
    X = data.drop("quality", axis=1)
    Y = data["quality"].apply(lambda y: 1 if y >= 7 else 0)  # Binarizing quality

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_resampled, Y_resampled, test_size=0.2, random_state=42
    )

    # Define XGBoost parameters
    xgb_params = {
        "objective": "binary:logistic",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    # Train the XGBoost model
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, Y_train)

    # Evaluate the model
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

    return model, accuracy, roc_auc, X.columns

# Streamlit Application
def main():
    st.title("🍷 Wine Quality Prediction")

    # Load and train the model
    model, accuracy, roc_auc, feature_names = train_model()

    # Display model evaluation metrics
    st.write("### Model Performance")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**ROC-AUC Score**: {roc_auc:.2f}")

    # Sidebar for user input
    st.sidebar.header("Input Wine Attributes")

    def user_input_features():
        user_data = {}
        data = load_data()
        for feature in feature_names:
            value = st.sidebar.slider(
                f"{feature}",
                float(data[feature].min()),
                float(data[feature].max()),
                float(data[feature].mean()),
            )
            user_data[feature] = value
        return pd.DataFrame(user_data, index=[0])

    input_data = user_input_features()

    # Make predictions
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0, 1]

        st.write("### Prediction")
        st.write(
            "✅ The wine is predicted to be **Good Quality**"
            if prediction == 1
            else "❌ The wine is predicted to be **Bad Quality**"
        )
        st.write(f"**Prediction Probability**: {prediction_proba:.2f}")

        # Display feature importance
        st.write("### Feature Importance")
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=feature_names, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
