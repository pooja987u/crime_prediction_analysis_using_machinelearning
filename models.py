import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Define paths for models and datasets
MODEL_PATH = "models/"
DATASET_PATH = "datasets/"

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

# Load models and scalers
scaler = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb'))
le_day = pickle.load(open(os.path.join(MODEL_PATH, 'le_day.pkl'), 'rb'))
le_crime = pickle.load(open(os.path.join(MODEL_PATH, 'le_crime.pkl'), 'rb'))
knn_model = pickle.load(open(os.path.join(MODEL_PATH, 'KNN.pkl'), 'rb'))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get user input from the form
            year = int(request.form["Year"])
            month = int(request.form["Month"])
            day_of_week = request.form["DayOfWeek"]
            hour = int(request.form["Hour"])
            latitude = float(request.form["Latitude"])
            longitude = float(request.form["Longitude"])

            # Encode Day of Week
            day_encoded = le_day.transform([day_of_week])[0]

            # Prepare input features
            features = np.array([[year, month, day_encoded, hour, latitude, longitude]])

            # Scale the features
            features_scaled = scaler.transform(features)

            # Make a prediction
            crime_type_encoded = knn_model.predict(features_scaled)[0]
            
            # Decode prediction to original crime type
            crime_type = le_crime.inverse_transform([crime_type_encoded])[0]

            return render_template("predict_future.html", prediction=f"Predicted Crime Type: {crime_type}")

        except Exception as e:
            return render_template("predict_future.html", prediction=f"Error: {e}")

    return render_template("predict_future.html")


if __name__ == "__main__":
    app.run(debug=True)
