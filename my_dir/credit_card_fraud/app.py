from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__)

# Define paths
base_save_dir = r'C:\Users\shamb\Desktop\Ai powered credit card fraud detection\my_dir'
model_path_h5 = os.path.join(base_save_dir, 'credit_card_fraud_model.h5')
model_path_keras = os.path.join(base_save_dir, 'credit_card_fraud_model.keras')
scaler_path = os.path.join(base_save_dir, 'saved_model', 'model', 'scaler.joblib')

# Load the model (choose the appropriate path)
try:
    if os.path.exists(model_path_h5):
        model = tf.keras.models.load_model(model_path_h5)
    elif os.path.exists(model_path_keras):
        model = tf.keras.models.load_model(model_path_keras)
    else:
        raise FileNotFoundError("Model file not found")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit()

# Load the saved scaler
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading scaler: {e}")
    traceback.print_exc()
    exit()

@app.route('/')
def home():
    return "Credit Card Fraud Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        
        # Ensure categorical columns are present
        categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
            else:
                return jsonify({'error': f'Missing column: {col}'})
        
        # Scale the data
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)
        prediction = (prediction > 0.5).astype(int)
        
        return jsonify({'prediction': int(prediction[0][0])})
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
