import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Visualization Function
def visualize(dataset):
    print("\nVisualization : Fraud vs Non-fraud transactions")
    sns.countplot(data=dataset, x="Fraud", hue="Fraud", palette=["Green", "Red"])
    plt.title("Fraud vs Non-fraud transactions")
    plt.xlabel("Fraud (0 = Non-fraud, 1 = Fraud)")
    plt.show()

    print("\nVisualization : Fraud Distribution by Transaction Type")
    sns.countplot(data=dataset, x="TransactionType", hue="Fraud", palette=["Green", "Red"])
    plt.title("Fraud Distribution by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.show()

# Model Training & Evaluation
def preprocess_and_train(dataset):
    # Encode categorical columns
    categorical_columns = ["TransactionType", "Location", "DeviceType", "TimeOfDay"]
    dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
    print("\nDataset after encoding categorical columns:\n", dataset.head())

    # Define features (X) and target (y)
    X = dataset.drop("Fraud", axis=1)
    y = dataset["Fraud"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("\nClass distribution after SMOTE:\n", y_train_resampled.value_counts())

    # Random Forest Model
    rf = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42
    )

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        rf, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_rf = grid_search.best_estimator_
    print("\nBest Parameters for Random Forest:", grid_search.best_params_)

    # Save the model
    joblib.dump(best_rf, "fraud_model.pkl")
    print("\nModel saved successfully as 'fraud_model.pkl'")

    # Predictions
    rf_pred = best_rf.predict(X_test)

    # Evaluate model
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)

    # Display metrics
    show_results(rf_acc, rf_prec, rf_rec, rf_f1, y_test, rf_pred)

# Display Model Results
def show_results(rf_acc, rf_prec, rf_rec, rf_f1, y_test, rf_pred):
    print("\nRandom Forest Performance")
    print(f"Accuracy: {rf_acc*100:.2f}")
    print(f"Precision: {rf_prec*100:.2f}")
    print(f"Recall: {rf_rec*100:.2f}")
    print(f"F1 Score: {rf_f1*100:.2f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))

# Data Loading & Cleaning
def load_and_clean_data(dataset):
    print("\nFirst five entries:\n", dataset.head())
    print("\nMissing values:\n", dataset.isnull().sum())

    # Fill missing values
    dataset["Amount"] = dataset["Amount"].fillna(dataset["Amount"].mean())
    print("\nMissing values after imputation:\n", dataset.isnull().sum())

    # Drop unnecessary columns
    dataset.drop(columns=["TransactionID", "CustomerID"], inplace=True)

    # Identify numeric columns for outlier detection
    numeric_columns = ["Amount", "TransactionSpeed"]

    # Boxplot BEFORE Outlier Removal
    print("\nBoxplot Before Outlier Removal:")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=dataset[numeric_columns])
    plt.title("Boxplot of Numeric Features (Before Outlier Removal)")
    plt.xlabel("Feature Name")
    plt.ylabel("Value Range")
    plt.show()

    # Remove outliers using IQR
    for col in numeric_columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    print("\nDataset after outlier removal:\n", dataset.head())

    # Boxplot AFTER Outlier Removal
    print("\nBoxplot After Outlier Removal:")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=dataset[numeric_columns])
    plt.title("Boxplot of Numeric Features (After Outlier Removal)")
    plt.xlabel("Feature Name")
    plt.ylabel("Value Range")
    plt.show()

    visualize(dataset)
    preprocess_and_train(dataset)

# Main Function
def main():
    print("Fraud Detection Case Study")
    dataset = pd.read_csv("fraud_detection.csv")
    load_and_clean_data(dataset)

if __name__ == "__main__":
    main()
