
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("Mental_Health_Model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            input_data = {
                "Study_Hours": float(form_data["study_hours"]),
                "Age": int(form_data["age"]),
                "Avg_Daily_Usage_Hours": float(form_data["social_hours"]),
                "Daily_Unlocks": int(form_data["unlocks"]),
                "Physical_Activity_Hours": float(form_data["activity"]),
                "Sleep_Hours_Per_Night": float(form_data["sleep"]),
                "Stress_Level": form_data["stress"],
                "Gender": form_data["gender"],
                "Academic_Level": form_data["academic_level"],
                "Most_Used_Platform": form_data["platform"],
                "Purpose_Of_Use": form_data["purpose"],
                "Grouped_country": form_data["country"]
            }

            # Make prediction using the trained model
            X = pd.DataFrame([input_data])
            prediction = round(float(model.predict(X)[0]), 2)

        except (ValueError, KeyError) as error:
            prediction = None
            print("Input error:", error)

    return render_template(
        "index.html",
        prediction=prediction,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
