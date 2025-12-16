"""Model training and tracking for credit risk model."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
from data_processing import prepare_training_data
from target_engineering import prepare_full_dataset
import warnings
warnings.filterwarnings('ignore')


def prepare_model_data():
    """Load and prepare data for modeling."""
    # Load raw data and create features + target
    from data_processing import load_data
    df = load_data("data/raw/data.csv")
    full_data = prepare_full_dataset(df)
    
    # Features and target
    X = full_data[['recency', 'frequency', 'monetary_total']]
    y = full_data['is_high_risk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Class distribution in train: {pd.Series(y_train).value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression."""
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        # Model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log to MLflow
        mlflow.log_params({
            'model_type': 'logistic_regression',
            'random_state': 42,
            'max_iter': 1000,
            'features': 'recency,frequency,monetary_total'
        })
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        
        print("Logistic Regression Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")
        
        return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest with hyperparameter tuning."""
    with mlflow.start_run(run_name="random_forest_tuned"):
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_params({
            'model_type': 'random_forest',
            'random_state': 42,
            'cv_folds': 3
        })
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.log_metric('best_cv_score', grid_search.best_score_)
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        print("Random Forest Results:")
        print(f"  Best params: {grid_search.best_params_}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")
        
        return best_model, metrics


def compare_models(logreg_metrics, rf_metrics):
    """Compare model performance and select best."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison = pd.DataFrame({
        'Logistic Regression': logreg_metrics,
        'Random Forest': rf_metrics
    }).T
    
    print(comparison.round(3))
    
    # Select best by ROC-AUC
    if logreg_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model_name = 'Logistic Regression'
        print(f"\n✅ Best model: {best_model_name} (ROC-AUC: {logreg_metrics['roc_auc']:.3f})")
    else:
        best_model_name = 'Random Forest'
        print(f"\n✅ Best model: {best_model_name} (ROC-AUC: {rf_metrics['roc_auc']:.3f})")
    
    return best_model_name


def main():
    """Main training pipeline."""
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit_risk_modeling")
    
    print("Starting model training pipeline...")
    print("="*50)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_model_data()
    
    print("\nTraining models...")
    print("-"*30)
    
    # Train models
    logreg_model, logreg_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    
    print()
    
    rf_model, rf_metrics = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    # Compare and select best
    best_model_name = compare_models(logreg_metrics, rf_metrics)
    
    print("\n✅ Training complete!")
    print(f"MLflow UI: run 'mlflow ui' and visit http://localhost:5000")


if __name__ == "__main__":
    main()