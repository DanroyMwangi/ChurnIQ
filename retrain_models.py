#!/usr/bin/env python3
"""
Script to retrain the churn prediction models with current scikit-learn version
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from joblib import dump
import os

def load_and_prepare_data():
    """Load and prepare the churn data"""
    print("Loading data...")
    df = pd.read_csv('final_churn_data.csv')
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop the unnamed index column and VisitorID if they exist
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'VisitorID' in df.columns:
        df = df.drop('VisitorID', axis=1)
    
    print(f"Dataset after cleaning: {df.shape}")
    print(df.head())
    
    return df

def create_churn_labels(df):
    """Create churn labels based on business logic"""
    print("Creating churn labels...")
    
    # Define thresholds for churn prediction
    # These thresholds should be based on your business understanding
    time_since_last_visit_threshold = 180  # days
    avg_time_between_visits_threshold = 120  # days
    low_frequency_threshold = 5  # purchases
    negative_profit_threshold = 0  # profit less than this
    
    # Create churn label based on multiple criteria
    churn_conditions = (
        (df['time_since_last_visit'] > time_since_last_visit_threshold) |
        (df['average_time_btn_visits'] > avg_time_between_visits_threshold) |
        (df['FrequencyOfPurchases'] < low_frequency_threshold) |
        (df['Profit'] < negative_profit_threshold)
    )
    
    df['churn'] = churn_conditions.astype(int)
    
    print(f"Churn distribution:")
    print(df['churn'].value_counts())
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    return df

def train_models(df):
    """Train both linear regression and random forest models"""
    print("Preparing features and targets...")
    
    # Features for prediction
    feature_columns = ['AverageSpendPerVisitKsh', 'Profit', 'FrequencyOfPurchases', 
                      'average_time_btn_visits', 'time_since_last_visit']
    
    X = df[feature_columns]
    y_classification = df['churn']  # For RandomForest (classification)
    
    # For linear regression, we'll predict a continuous churn score
    # based on normalized risk factors
    y_regression = (
        (df['time_since_last_visit'] / df['time_since_last_visit'].max()) * 0.3 +
        (df['average_time_btn_visits'] / df['average_time_btn_visits'].max()) * 0.3 +
        ((df['FrequencyOfPurchases'].max() - df['FrequencyOfPurchases']) / df['FrequencyOfPurchases'].max()) * 0.2 +
        ((df['Profit'].max() - df['Profit']) / (df['Profit'].max() - df['Profit'].min())) * 0.2
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Features: {feature_columns}")
    
    # Split the data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_classification, y_regression, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    print("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_reg_train)
    
    # Evaluate Linear Regression
    y_reg_pred = linear_model.predict(X_test_scaled)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    print(f"Linear Regression MSE: {mse:.4f}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(X_train_scaled, y_class_train)
    
    # Evaluate Random Forest
    y_class_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return linear_model, rf_model, scaler

def save_models(linear_model, rf_model, scaler):
    """Save the trained models and scaler"""
    print("Saving models...")
    
    # Save models with the same names as the original files
    dump(linear_model, 'churn-prediction-multiple-features.joblib')
    dump(rf_model, 'rf_model.joblib')
    dump(scaler, 'scaler-multiple-features.joblib')
    
    print("Models saved successfully!")
    print("- churn-prediction-multiple-features.joblib (Linear Regression)")
    print("- rf_model.joblib (Random Forest)")
    print("- scaler-multiple-features.joblib (StandardScaler)")

def main():
    """Main function to retrain all models"""
    print("=== ChurnIQ Model Retraining ===")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Create churn labels
        df = create_churn_labels(df)
        
        # Train models
        linear_model, rf_model, scaler = train_models(df)
        
        # Save models
        save_models(linear_model, rf_model, scaler)
        
        print("\n=== Retraining Complete ===")
        print("Your Flask application should now work with the updated models!")
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        raise

if __name__ == "__main__":
    main()
