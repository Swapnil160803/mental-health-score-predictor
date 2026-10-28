
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# Project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

# Initialize Flask
app = Flask(__name__)

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    input_data = None

    if request.method == "POST":
        try:
            # Collect user inputs using the model's feature names
            input_data = pd.DataFrame([{
                "Study_Hours": float(request.form["study_hours"]),
                "Age": int(request.form["age"]),
                "Avg_Daily_Usage_Hours": float(
                    request.form["usage_hours"]
                ),
                "Daily_Unlocks": int(request.form["unlocks"]),
                "Physical_Activity_Hours": float(
                    request.form["activity"]
                ),
                "Sleep_Hours_Per_Night": float(
                    request.form["sleep"]
                ),
                "Stress_Level": request.form["stress"],
                "Gender": request.form["gender"],
                "Academic_Level": request.form["academic"],
                "Most_Used_Platform": request.form["platform"],
                "Purpose_Of_Use": request.form["purpose"],
                "Grouped_country": request.form["country"]
            }])

            # Predict using the trained model
            result = model.predict(input_data)[0]
            prediction = round(float(result), 2)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
