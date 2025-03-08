import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

data = pd.read_csv("disaster_data.csv")  # Ensure this file is present
features = ["year", "location", "disaster_type", "disaster_magnitude", "weather", "total_population"]
X = data[features]
scaler = MinMaxScaler()
scaler.fit(X)  # Fit scaler with dataset

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        sample_input = np.array(data["features"]).reshape(1, -1)
        sample_scaled = scaler.transform(sample_input)

        # Predict using the model
        prediction = model.predict(sample_scaled).tolist()

        return jsonify({"predicted_resources": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
