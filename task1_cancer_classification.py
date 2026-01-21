import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_pipeline(data_path, test_size, model_type):
    # 1. Load Data
    # Note: UCI Wisconsin Diagnostic usually has no header; check your CSV format
    df = pd.read_csv(data_path)
    
    # Basic cleaning: Drop ID and empty columns if they exist
    df = df.dropna(axis=1, how='all')
    
    # Assuming standard UCI format: Col 0 is ID, Col 1 is Diagnosis (M/B)
    X = df.iloc[:, 2:].values 
    y = df.iloc[:, 1].map({'M': 1, 'B': 0}).values

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 3. Preprocessing (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Model Selection
    if model_type == 'logistic':
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier(max_depth=5)

    model.fit(X_train, y_train)

    # 5. Evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_err = 1 - accuracy_score(y_train, y_train_pred)
    test_err = 1 - accuracy_score(y_test, y_test_pred)

    print(f"--- Results for {model_type.upper()} ---")
    print(f"Train Error: {train_err:.4f}")
    print(f"Test Error:  {test_err:.4f}")
    print(f"Generalization Gap: {abs(train_err - test_err):.4f}")
    print("-" * 30)
    print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_test_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model", type=str, choices=['logistic', 'tree'], default='logistic')
    args = parser.parse_args()
    
    run_pipeline(args.data, args.test_size, args.model)