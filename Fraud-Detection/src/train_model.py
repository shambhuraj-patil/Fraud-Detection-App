

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# Function to load and clean the dataset
def load_and_clean_data(dataset):
    # Load the data
    print("\nFirst five entries from loaded dataset :\n",dataset.head())

    # Check for missing values
    print("\nMissing values in the dataset :\n",dataset.isnull().sum())

    # Fill missing values in 'Amount' column with the mean value
    dataset["Amount"] = dataset["Amount"].fillna(dataset["Amount"].mean())
    print("\nMissing values after filling mean :\n",dataset.isnull().sum())

    # Check class distribution
    print("\nFraud Class Distribution Before Handling Imbalance :\n",dataset["Fraud"].value_counts())

    # Drop unnecessary columns
    dataset.drop(columns=["TransactionID","CustomerID"],inplace=True)
    print("\nDataset after dropping irrelevant columns :\n",dataset.head())

    # Check for outliers using a boxplot
    numeric_columns = ["Amount","TransactionSpeed"]
    sns.boxplot(data=dataset[numeric_columns])
    plt.title("Boxplot for numeric columns")
    plt.xlabel("Columns")
    plt.ylabel("Count")
    plt.show()

    # Removing outliers using the IQR (Interquartile Range) method
    for col in numeric_columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    print("\nDataset after removing outliers :\n",dataset.head())
    return dataset

# Function to visualize fraud distribution
def visualize(dataset):
    print("\nVisualization : Fraud vs Non-fraud transactions")
    sns.countplot(data=dataset,x="Fraud",hue="Fraud",palette=["Green","Red"])
    plt.title("Fraud vs Non-fraud transactions")
    plt.xlabel("Fraud (0 = Non-fraud, 1 = Fraud)")
    plt.show()

    print("\nVisualization : Fraud Distribution by Transaction Type")
    sns.countplot(data=dataset,x="TransactionType",hue="Fraud",palette=["Green","Red"])
    plt.title("Fraud Distribution by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.show()

# Function to preprocess the dataset
def preprocess_data(dataset):
    # Encode categorical columns 
    categorical_columns = ["TransactionType","Location","DeviceType","TimeOfDay"]
    dataset = pd.get_dummies(dataset,columns=categorical_columns,drop_first=True)
    print("\nDataset after encoding categorical columns :\n",dataset.head())

    # Define features (X) and target (y)
    x = dataset.drop("Fraud",axis=1)
    y = dataset["Fraud"]
    
    # Split dataset into training (70%) and testing (30%) sets
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    
    # Display class distribution after resampling
    print("\nClass distribution after resampling:\n", y_train_resampled.value_counts())

    # Standardize features for better performance
    sc = StandardScaler()
    scaled_x_train = sc.fit_transform(x_train_resampled)
    scaled_x_test = sc.transform(x_test)

    # Train Logistic Regression model
    lr = LogisticRegression()
    lr.fit(scaled_x_train,y_train_resampled)
    lr_prediction = lr.predict(scaled_x_test)

    # Evaluate Logistic Regression model
    lr_acc = accuracy_score(y_test, lr_prediction)
    lr_prec = precision_score(y_test, lr_prediction)
    lr_rec = recall_score(y_test, lr_prediction)
    lr_f1 = f1_score(y_test, lr_prediction)

    # Define hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Train Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Perform hyperparameter tuning using Grid Search
    grid_search = GridSearchCV(rf,param_grid,cv=5,n_jobs=-1)
    grid_search.fit(x_train_resampled,y_train_resampled)

    # Get the best hyperparameters
    best_rf = grid_search.best_estimator_
    print("\nBest Parameters for Random Forest:", grid_search.best_params_)

    # Make predictions using the best Random Forest model
    rf_prediction = best_rf.predict(x_test)

    # Evaluate Random Forest model
    rf_acc = accuracy_score(y_test, rf_prediction)
    rf_prec = precision_score(y_test, rf_prediction)
    rf_rec = recall_score(y_test, rf_prediction)
    rf_f1 = f1_score(y_test, rf_prediction)

    # Save best model
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)

    # Save the fitted scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(sc, f)

    print("\nModel and Scaler saved as 'fraud_model.pkl' and 'scaler.pkl'")

    return lr_acc, lr_prec, lr_rec, lr_f1, rf_acc, rf_prec, rf_rec, rf_f1

# Function to display model evaluation results
def results(lr_acc, lr_prec, lr_rec, lr_f1, rf_acc, rf_prec, rf_rec, rf_f1):
    print("\nLogistic Regression Performance:")
    print(f"Accuracy: {lr_acc*100:.2f}")
    print(f"Precision: {lr_prec*100:.2f}")
    print(f"Recall: {lr_rec*100:.2f}")
    print(f"F1 Score: {lr_f1*100:.2f}")

    print("\nRandom Forest Performance:")
    print(f"Accuracy: {rf_acc*100:.2f}")
    print(f"Precision: {rf_prec*100:.2f}")
    print(f"Recall: {rf_rec*100:.2f}")
    print(f"F1 Score: {rf_f1*100:.2f}")

# Main function to execute the fraud detection pipeline
def main():
    print("Fraud Detection Case Study")
    dataset = pd.read_csv("fraud_detection/fraud_detection.csv")
    dataset = load_and_clean_data(dataset)
    visualize(dataset)
    lr_acc, lr_prec, lr_rec, lr_f1, rf_acc, rf_prec, rf_rec, rf_f1 = preprocess_data(dataset)
    results(lr_acc, lr_prec, lr_rec, lr_f1, rf_acc, rf_prec, rf_rec, rf_f1)
    
if __name__ == "__main__":
    main()
