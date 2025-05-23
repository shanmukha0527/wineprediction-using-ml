# 🍷 Wine Quality Prediction App

This Streamlit web application predicts the quality of red wine using machine learning. It classifies wine as "Good Quality" or "Bad Quality" based on various physicochemical properties.

## Features

- Loads and preprocesses the red wine dataset (`winequality-red.csv`)
- Binarizes the wine quality score (Good: quality ≥ 7, Bad: quality < 7)
- Handles class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- Trains a classifier using XGBoost
- Evaluates performance using accuracy and ROC-AUC
- Accepts user input for prediction through a sidebar
- Displays predicted result and feature importance plot

## Requirements

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn xgboost imbalanced-learn
