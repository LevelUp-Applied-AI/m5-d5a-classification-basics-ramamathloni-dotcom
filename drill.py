"""
Module 5 Week A — Core Skills Drill: Classification & Evaluation Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """Split a DataFrame into train and test sets with stratification.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features (X) and target (y), then split with stratification
    """Split a DataFrame into train and test sets with stratification."""
    # Separate the features (X) and the target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Perform the split, using stratify=y to ensure class proportions are 
    # consistent across both training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Array of true labels (0 or 1).
        y_pred: Array of predicted labels (0 or 1).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
        Values are floats.
    """
    # TODO: Compute all four metrics using scikit-learn functions
    """Compute classification metrics from true and predicted labels."""
    
    # We use scikit-learn functions to compare the actual values (y_true)
    # with what the model predicted (y_pred).
    
    # Accuracy: How many predictions did we get right overall?
    # Precision: When the model predicts 'churn', how often is it actually correct?
    # Recall: Out of all the people who actually churned, how many did we catch?
    # F1-score: The harmonic mean that balances Precision and Recall.
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """Run stratified k-fold cross-validation with LogisticRegression.

    Args:
        X_train: Training features (numeric only).
        y_train: Training labels.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dictionary with keys: 'scores' (array of fold scores),
        'mean' (float), 'std' (float).
    """
    # TODO: Create a LogisticRegression model and run cross_val_score
    """Run stratified k-fold cross-validation with LogisticRegression."""
    
    # 1. Initialize the model
    # We use class_weight="balanced" because in churn problems, 
    # the minority class (churners) is what we care about most.
    # This parameter tells the model to penalize mistakes on the minority class more.
    model = LogisticRegression(
        random_state=random_state, 
        max_iter=1000, 
        class_weight="balanced"
    )
    
    # 2. Run cross-validation
    # cross_val_score splits the data into 'n_folds' parts automatically
    # and calculates the accuracy for each fold.
    scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="accuracy")
    
    # 3. Calculate statistics
    # mean: The average performance across all folds.
    # std: The standard deviation, which tells us how "bouncy" or stable the model is.
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/telecom_churn.csv")
    print(f"Loaded {len(df)} rows")

    # Task 1: Split
    numeric_cols = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen", "has_partner",
                    "has_dependents"]
    df_numeric = df[numeric_cols + ["churned"]]

    result = split_data(df_numeric)
    if result is not None:
        X_train, X_test, y_train, y_test = result
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Task 2: Metrics
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        if metrics:
            print(f"Metrics: {metrics}")

        # Task 3: Cross-validation
        cv_results = run_cross_validation(X_train, y_train)
        if cv_results:
            print(f"CV: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")
