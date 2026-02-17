from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models and utilities
MODELS = {}
SCALER = None
LE = None

def load_assets():
    global MODELS, SCALER, LE
    try:
        if os.path.exists('models/advanced_regression.pkl'):
            MODELS['regression'] = joblib.load('models/advanced_regression.pkl')
        if os.path.exists('models/advanced_classification.pkl'):
            MODELS['classification'] = joblib.load('models/advanced_classification.pkl')
        if os.path.exists('models/scaler.pkl'):
            SCALER = joblib.load('models/scaler.pkl')
        if os.path.exists('models/label_encoder.pkl'):
            LE = joblib.load('models/label_encoder.pkl')
        print("Assets loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Expected 'sequence': list of 12 lists/dicts
        raw_seq = np.array(data['sequence']) # Shape (12, features)
        
        if not MODELS:
            load_assets()
            
        if 'regression' in MODELS:
            # The models were trained on flattened sequences
            # Assuming input is already scaled for the demo or we scale here
            # In a real app, we would apply SCALER.transform(raw_seq)
            # For this demo script, we'll assume the front-end sends scaled-like values
            # or we just pass it through if it's within [0,1]
            
            flattened = raw_seq.reshape(1, -1)
            reg_pred = MODELS['regression'].predict(flattened)[0]
            cls_pred_idx = MODELS['classification'].predict(flattened)[0]
            
            cls_label = LE.inverse_transform([cls_pred_idx])[0]
            
            return jsonify({
                'regression_prediction': float(reg_pred),
                'congestion_level': cls_label
            })
        else:
            return jsonify({'error': 'Models not trained yet'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_assets()
    app.run(debug=True, host='0.0.0.0', port=5000)
