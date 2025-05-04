from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from dairy_model_test_prediction import (
    prepareDataforPrediction,
    model_prediction,
)

app = Flask(__name__)
CORS(app)


# Load the dataset (replace 'your_dataset.csv' with the actual path to your dataset)
dataset_path = "C:/Users/alexa/OneDrive/Anexos/Fiap/stocksage_fiap_challenge/stocksage_backend/dairy/dairy_data/current_processed_data.csv"
df = pd.read_csv(dataset_path)


@app.route("/api/data", methods=["GET"])
def get_unique_products():
    # Convert the DataFrame to a dictionary format suitable for JSON
    data = df.to_dict(orient="records")
    # data = df.tail(1000).to_dict(orient="records")

    return jsonify({"data": data})


@app.route("/api/prediction", methods=["POST"])
def prediction():
    data = request.get_json()  # Parse JSON data from the request

    if not data:
        return jsonify({"error": "Dados não fornecidos"}), 400

    # Prepare data for prediction
    data_prepared = prepareDataforPrediction(data)

    # Perform prediction
    predict = model_prediction(data_prepared)

    # Convert numpy types to native Python types
    data_prepared = {
        key: (
            int(value)
            if isinstance(value, np.integer)
            else float(value) if isinstance(value, np.floating) else value
        )
        for key, value in data_prepared.items()
    }

    # If predict is a numpy array, convert it to a Python list
    if isinstance(predict, np.ndarray):
        predict = predict.tolist()
    return (
        jsonify(
            {
                "received_data": data,
                "data_prepared": data_prepared,
                "prediction": predict,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
