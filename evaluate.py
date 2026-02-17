import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(task_type='regression'):
    if task_type == 'regression':
        X = np.load('data/X_reg.npy')
        y = np.load('data/y_reg.npy')
        model_path = f'models/advanced_{task_type}.pkl'
        is_cls = False
    else:
        X = np.load('data/X_cls.npy')
        y = np.load('data/y_cls.npy')
        model_path = f'models/advanced_{task_type}.pkl'
        is_cls = True
        le = joblib.load('models/label_encoder.pkl')

    X_flattened = X.reshape(X.shape[0], -1)
    split = int(0.8 * len(X))
    X_test = X_flattened[split:]
    y_test = y[split:]

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping evaluation.")
        return

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    if not is_cls:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"--- {task_type.upper()} STATS ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[:100], label='Actual', alpha=0.7)
        plt.plot(preds[:100], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted - {task_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'notebooks/{task_type}_eval.png')
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"--- {task_type.upper()} STATS ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
        plt.title(f'Confusion Matrix - {task_type}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'notebooks/{task_type}_cm.png')
    
    plt.close()

if __name__ == "__main__":
    evaluate_models('regression')
    evaluate_models('classification')
    print("Evaluation complete. Results saved in notebooks/")
