from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the models
compressive_model = joblib.load('compressive_model.pkl')
tensile_model = joblib.load('tensile_model.pkl')

# Define column names
COLUMN_NAMES = [
    'Alkaline_Liquid_Ratio',
    'Fly_Ash (kg/m3)',
    'NaOH (kg/m3)(1 part)',
    'Na2SiO3(kg/m3)(2 part)',
    'Copper_Slag (kg/m3)',
    'Total_Coarse_Aggregate (kg/m3)',
    'RCA(Percentage)',
    'RCA(kg/m3)',
    'Natural_Aggregate(kg/m3)',
    'Curing_Method',
    'Curing_Time (Days)',
    'Slump (mm)'
]

@app.route('/')
def home():
    return "ML API is Running Successfully 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')

        if not features:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        input_values = [features[col] for col in COLUMN_NAMES]
        input_df = pd.DataFrame([input_values], columns=COLUMN_NAMES)

        compressive = compressive_model.predict(input_df)[0]
        tensile = tensile_model.predict(input_df)[0]

        return jsonify({
            "compressive_strength": round(float(compressive), 2),
            "tensile_strength": round(float(tensile), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # For cloud platforms like Render, use PORT from environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
