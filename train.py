from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

def train_and_save_models(task_type='regression'):
    if task_type == 'regression':
        X = np.load('data/X_reg.npy')
        y = np.load('data/y_reg.npy')
        is_cls = False
    else:
        X = np.load('data/X_cls.npy')
        y = np.load('data/y_cls.npy')
        is_cls = True
        
    # Flatten the 12x9 sequence into 108 features for MLP
    X_flattened = X.reshape(X.shape[0], -1)
    
    split = int(0.8 * len(X))
    X_train, X_test = X_flattened[:split], X_flattened[split:]
    y_train, y_test = y[:split], y[split:]
    
    if not is_cls:
        # Baseline: Smaller MLP
        print(f"Training Baseline MLP for {task_type}...")
        baseline = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
        baseline.fit(X_train, y_train)
        
        # Advanced: Larger MLP with Attention-like features (e.g., deeper/wider)
        print(f"Training Advanced MLP for {task_type}...")
        advanced = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        advanced.fit(X_train, y_train)
    else:
        # Classification
        print(f"Training Baseline MLP for {task_type}...")
        baseline = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
        baseline.fit(X_train, y_train)
        
        print(f"Training Advanced MLP for {task_type}...")
        advanced = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        advanced.fit(X_train, y_train)
        
    # Save models
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(baseline, f'models/baseline_{task_type}.pkl')
    joblib.dump(advanced, f'models/advanced_{task_type}.pkl')
    
    # Evaluation
    preds = advanced.predict(X_test)
    if not is_cls:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{task_type.capitalize()} Advanced - RMSE: {rmse:.4f}")
    else:
        acc = accuracy_score(y_test, preds)
        print(f"{task_type.capitalize()} Advanced - Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_and_save_models('regression')
    train_and_save_models('classification')
    print("Training complete. Models saved as .pkl files.")
