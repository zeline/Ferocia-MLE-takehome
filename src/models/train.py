"""
Model training script for term deposit prediction.

This script trains and evaluates a classification model, then saves
the artifacts for serving.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import preprocess_for_training, save_preprocessor

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'logistic') -> object:
    """
    train a classification model.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training labels
        model_type: 'logistic' or 'random_forest'
        
    Returns:
        Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',  # Handle imbalanced classes
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model and print metrics.
    
    Args:
        model: Trained model
        X_test: Preprocessed test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n" + "-" * 50)
    print("Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    print("-" * 50)
    print("Confusion Matrix:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                No    Yes")
    print(f"Actual No     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Yes    {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    return metrics

def save_model(model, filepath: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")

def main():
    """
    Main training pipeline."""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'DATA' / 'dataset.csv'
    artifacts_dir = project_root / 'artifacts'
    
    print("=" * 50)
    print("TERM DEPOSIT PREDICTION - MODEL TRAINING")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print(f"\n📊 Loading and preprocessing data from {data_path}")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(str(data_path))
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution (train): {np.bincount(y_train)} (No/Yes)")
    
    # Step 2: Train model
    print("\n🔧 Training Logistic Regression model...")
    model = train_model(X_train, y_train, model_type='logistic')
    
    # Step 3: Evaluate model
    print("\n📈 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 4: Save artifacts
    print("\n💾 Saving artifacts...")
    save_model(model, str(artifacts_dir / 'model.pkl'))
    save_preprocessor(preprocessor, str(artifacts_dir / 'preprocessor.pkl'))
    
    print("\n" + "=" * 50)
    print("✅ Training complete!")
    print("=" * 50)
    print(f"\nArtifacts saved to: {artifacts_dir}/")
    print("  - model.pkl")
    print("  - preprocessor.pkl")
    
    # Notes for improvement
    print("\n" + "-" * 50)
    print("📝 NOTES FOR FUTURE IMPROVEMENT:")
    print("-" * 50)
    print("""
    1. Try other models: Random Forest, XGBoost, LightGBM
    2. Feature selection to reduce dimensionality
    3. Cross-validation for more robust evaluation
    4. Future analysis into relationships between features
    5. Removing 'duration' feature (leaky - unknown before call)

    """)
    
    return model, preprocessor, metrics

if __name__ == '__main__':
    model, preprocessor, metrics = main()