# Traffic Congestion Prediction using Sequential Deep Learning

This project implements a sequential neural intelligence system to predict future traffic volume and classify congestion levels based on historical data.

## Project Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Evaluation plots and visualizations.
- `models/`: Saved model weights (`.pkl`) and encoders.
- `utils/`: 
    - `generate_data.py`: Synthetic traffic data generation.
    - `preprocess.py`: Feature engineering and sequence creation.
- `train.py`: Training script for baseline and advanced neural models.
- `evaluate.py`: Performance metrics and visualization script.
- `app.py`: Flask deployment backend.
- `templates/`: Interactive dashboard UI.

## Methodology
1. **Preprocessing**:
    - Converts timestamps to datetime and sorts chronologically.
    - Derived cyclical features (Hour Sin/Cos) and weekend indicators.
    - Normalizes features using `MinMaxScaler`.
    - Creates sliding window sequences (12-hour lookback).
2. **Architecture**:
    - **Baseline**: 64-unit Neural Network with Dropout.
    - **Advanced**: Multi-layer Sequential Neural Controller (128-64 units).
3. **Deployment**:
    - Predictive REST API built with Flask.
    - Modern Glassmorphism dashboard for real-time testing.

## How to Run
1. **Generate Data**: `python utils/generate_data.py`
2. **Preprocess**: `python utils/preprocess.py`
3. **Train**: `python train.py`
4. **Deploy**: `python app.py`

Navigate to `http://localhost:5000` to interact with the dashboard.
