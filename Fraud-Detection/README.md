# 🛡️ Credit Card Fraud Detection using Machine Learning

A machine learning-powered application that predicts fraudulent credit card transactions using real-world imbalanced data. The model is deployed with a user-friendly interface built using Streamlit.

> ⚠️ Fraud causes financial losses of billions globally every year. This tool helps detect anomalies in transactions to reduce fraud exposure.

---

## 📌 Live Demo

🎯 **Try the App Here**: [🔗 Streamlit App](https://ml-major-project-credit-fraud-check.streamlit.app/)  
📁 **View the Code**: [🔗 GitHub Repository](https://github.com/shambhuraj-patil/ML-Major-Project)

---

## 📊 Problem Statement

Credit card fraud detection is a critical challenge for financial systems. This project uses supervised machine learning to identify potentially fraudulent transactions using various features from anonymized transactional data.

---

## 🧠 ML Approach

- **Data Preprocessing**
  - Handled missing values
  - Removed outliers using **IQR**
  - Applied **One-Hot Encoding** on categorical data
- **Imbalanced Dataset**
  - Used **SMOTE** to synthetically balance fraud vs non-fraud cases
- **Models Used**
  - Logistic Regression (Baseline)
  - Random Forest Classifier (Final Model)
- **Evaluation Metrics**
  - Accuracy: **88%**
  - Precision & Recall for fraud class
  - Confusion Matrix

---

## 🛠 Tech Stack

| Tool / Library       | Purpose                      |
|----------------------|------------------------------|
| Python               | Core language                |
| Pandas, NumPy        | Data handling                |
| Scikit-learn         | ML models & preprocessing    |
| SMOTE (imblearn)     | Handle class imbalance       |
| Matplotlib, Seaborn  | Visualizations               |
| Streamlit            | Web app deployment           |
| Git & GitHub         | Version control & repository |

---

## 🚀 Deployment

The app is live using **Streamlit Cloud**, and also fully runnable locally.

### ✅ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/shambhuraj-patil/ML-Major-Project.git
cd ML-Major-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

ML-Major-Project/
│
├── app.py               # Streamlit app
├── fraud_detection.py   # Core ML model logic
├── dataset.csv          # Training data
├── requirements.txt     # Dependencies
└── README.md            # This file

---

## 👨‍💻 Author

Shambhuraj Patil
📍 Pune, India

📧 shambhurajpatil27@gmail.com

[LinkedIn](https://www.linkedin.com/in/shambhurajpatil/)

---

## 🙌 Feedback & Contribution
If you found this useful or want to collaborate, feel free to fork, raise an issue, or contribute!

## ⭐ If you like this project, please star the repo – it motivates and helps visibility!
