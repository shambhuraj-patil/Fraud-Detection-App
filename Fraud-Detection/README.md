# Fraud Detection System - Major ML Project

This project is a machine learning-based fraud detection system. It uses supervised learning models to identify fraudulent transactions based on historical transaction data. The system includes a data preprocessing pipeline, model training script, and an interactive Streamlit web application.

---

## Project Structure

- app.py # Streamlit Web Application
- fraud_detection.csv # Dataset for fraud detection
- fraud_model.pkl # Trained Random Forest model (or other chosen model)
- requirements.txt # Python dependencies for the project
- scaler.pkl # StandardScaler for input normalization
- train_model.py # Model training & preprocessing pipeline

---

## Features

- Machine learning model (Random Forest with GridSearchCV)
- Data preprocessing with outlier removal, encoding, scaling
- Handles class imbalance with SMOTE
- Streamlit web UI for interactive prediction
- CSV upload and fraud probability output
- Model evaluation: Accuracy, Precision, Recall, F1 Score

---

## How to Use

### Step 1: Train the Model

Run this in your terminal:

```bash
python train_model.py
```

- Loads fraud_detection.csv

- Cleans and preprocesses data

- Trains models (Logistic Regression & Random Forest)

- Saves fraud_model.pkl and scaler.pkl

### Step 2: Launch the App

```bash
streamlit run app.py
```
Then open your browser to the local Streamlit URL.
You can:

- Upload your CSV file

- Preprocess the data

- View fraud pattern visualizations

- Predict fraud using the trained model

- Download the results

---

## Input CSV Format
Your CSV file must contain the following columns:

TransactionID, CustomerID, Amount, TransactionType, 
Location, DeviceType, TimeOfDay, TransactionSpeed, Fraud

- You can download a sample file from within the app.

---

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

Or make sure you have:

- streamlit

- pandas

- scikit-learn

- matplotlib

- seaborn

- imblearn

---

## Deployment
This app is deployable on Streamlit Cloud:

- Upload your project repo to GitHub

- Go to Streamlit Cloud → “New App”

- Connect your GitHub repo

- Set app.py as the main file

- Deploy!

---

## Screenshots
![image](https://github.com/user-attachments/assets/0225051b-14c6-4183-a76e-ff108c26b489)

---

## Author
Shambhuraj

Major ML Project – Fraud Detection System

Built with ❤️ using Python, Streamlit, and Scikit-learn
