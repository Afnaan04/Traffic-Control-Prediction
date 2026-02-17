import pandas as pd
import numpy as np
import datetime

# Set seed for reproducibility
np.random.seed(42)

# Generate 30 days of hourly data
date_range = pd.date_range(start='2024-01-01', periods=24*30, freq='h')
data = []

for dt in date_range:
    # Basic seasonality: peak hours (8-9 AM, 5-6 PM)
    hour = dt.hour
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Base traffic volume depends on hour and weekend
    if 7 <= hour <= 9 or 16 <= hour <= 19:
        base_vol = 100 if is_weekend == 0 else 50
    elif 23 <= hour or hour <= 5:
        base_vol = 10
    else:
        base_vol = 40
        
    # Add random noise
    car_count = int(base_vol * np.random.uniform(0.8, 1.5))
    bus_count = int(base_vol * 0.1 * np.random.uniform(0.5, 1.2))
    truck_count = int(base_vol * 0.15 * np.random.uniform(0.4, 1.1))
    bike_count = int(base_vol * 0.05 * np.random.uniform(0.1, 2.0))
    
    total = car_count + bus_count + truck_count + bike_count
    
    # Congestion Level logic
    if total > 120:
        level = 'Jam'
    elif total > 80:
        level = 'High'
    elif total > 40:
        level = 'Normal'
    else:
        level = 'Low'
        
    data.append([dt, car_count, bus_count, truck_count, bike_count, total, level])

df = pd.DataFrame(data, columns=['Timestamp', 'CarCount', 'BusCount', 'TruckCount', 'BikeCount', 'Total', 'CongestionLevel'])
df.to_csv('data/traffic.csv', index=False)

print(f"Generated {len(df)} rows of synthetic traffic data in data/traffic.csv")
