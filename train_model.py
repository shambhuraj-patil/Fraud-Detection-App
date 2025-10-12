import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
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
    
    # Define hyperparameter grid for Random Forest (LOW LOAD)
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [10, 20],     
    }

    # Train Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Perform hyperparameter tuning using Grid Search
    grid_search = GridSearchCV(rf, param_grid, cv=2, n_jobs=4) 
    grid_search.fit(x_train_resampled, y_train_resampled) 

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

    print("\nModel saved as 'fraud_model.pkl'")

    # Return Random Forest metrics
    return rf_acc, rf_prec, rf_rec, rf_f1

# Function to display model evaluation results
def results(rf_acc, rf_prec, rf_rec, rf_f1):
    print("\nModel Performance (Random Forest):")
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
    rf_acc, rf_prec, rf_rec, rf_f1 = preprocess_data(dataset) 
    results(rf_acc, rf_prec, rf_rec, rf_f1)
    
if __name__ == "__main__":
    main()
