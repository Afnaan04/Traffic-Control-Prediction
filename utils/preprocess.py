import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Feature Engineering
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['IsWeekend'] = df['Timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical encoding for Hour
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Categorical Encoding for CongestionLevel
    le = LabelEncoder()
    df['CongestionLevel_Encoded'] = le.fit_transform(df['CongestionLevel'])
    
    # Numerical features to scale
    numerical_cols = ['CarCount', 'BusCount', 'TruckCount', 'BikeCount', 'Total', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsWeekend']
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save objects for deployment
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    return df, le, scaler

def create_sequences(data, target_col, window_size=12):
    X, y = [], []
    # Drop non-numerical and target columns for X
    feature_cols = ['CarCount', 'BusCount', 'TruckCount', 'BikeCount', 'Total', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsWeekend']
    
    data_values = data[feature_cols].values
    target_values = data[target_col].values
    
    for i in range(len(data) - window_size):
        X.append(data_values[i:i+window_size])
        y.append(target_values[i+window_size])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df, le, scaler = preprocess_data('data/traffic.csv')
    X_reg, y_reg = create_sequences(df, 'Total')
    X_cls, y_cls = create_sequences(df, 'CongestionLevel_Encoded')
    
    np.save('data/X_reg.npy', X_reg)
    np.save('data/y_reg.npy', y_reg)
    np.save('data/X_cls.npy', X_cls)
    np.save('data/y_cls.npy', y_cls)
    
    print("Preprocessing complete. Sequences saved in data/")
