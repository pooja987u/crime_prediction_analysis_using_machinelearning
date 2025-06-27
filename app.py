# Import necessary libraries
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

MODEL_PATH = "models/"
DATASET_PATH = "datasets/"
STATIC_PATH = "static/"

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'datasetfile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['datasetfile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        file_path = os.path.join(DATASET_PATH, file.filename)
        file.save(file_path)
        flash('Dataset uploaded successfully! You can now train the KNN model.')
        return render_template('upload.html')
    return render_template('upload.html')

@app.route('/train.html', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            dataset_files = os.listdir(DATASET_PATH)
            if not dataset_files:
                flash('No dataset found. Please upload a dataset first.')
                return redirect('/upload.html')

            dataset_path = os.path.join(DATASET_PATH, dataset_files[0])
            data = pd.read_csv(dataset_path)

            required_columns = ['Year', 'Month', 'DayOfWeek', 'Hour', 'Latitude', 'Longitude', 'CrimeType']
            if not all(col in data.columns for col in required_columns):
                flash(f"Dataset must contain the following columns: {', '.join(required_columns)}")
                return redirect('/upload.html')

            le_day = LabelEncoder()
            le_crime = LabelEncoder()
            data['DayOfWeek'] = le_day.fit_transform(data['DayOfWeek'])
            data['CrimeType'] = le_crime.fit_transform(data['CrimeType'])

            X = data[['Year', 'Month', 'DayOfWeek', 'Hour', 'Latitude', 'Longitude']]
            y = data['CrimeType']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            pickle.dump(scaler, open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'wb'))
            pickle.dump(le_day, open(os.path.join(MODEL_PATH, 'le_day.pkl'), 'wb'))
            pickle.dump(le_crime, open(os.path.join(MODEL_PATH, 'le_crime.pkl'), 'wb'))

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)

            acc = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)

            pickle.dump(knn, open(os.path.join(MODEL_PATH, 'KNN.pkl'), 'wb'))

            # Save confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_crime.classes_, yticklabels=le_crime.classes_)
            plt.title('Confusion Matrix - KNN')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(STATIC_PATH, 'confusion_matrix_KNN.png'))
            plt.close()

            flash(f'Model trained successfully with an accuracy of {acc:.2f}!')
            return redirect('/train_results')

        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect('/train.html')

    return render_template('train.html')


@app.route('/train_results')
def train_results():
    try:
        # Check if the model exists
        if not os.path.exists(os.path.join(MODEL_PATH, 'KNN.pkl')):
            flash('No trained model found. Please train the model first.')
            return redirect('/train')

        # Load the trained model
        with open(os.path.join(MODEL_PATH, 'KNN.pkl'), 'rb') as f:
            knn = pickle.load(f)

        # You can render some results or metrics here, for example, accuracy or other information
        accuracy = None
        with open(os.path.join(STATIC_PATH, 'confusion_matrix_KNN.png'), 'rb') as f:
            accuracy = f"Model is trained and confusion matrix is available."

        return render_template('train_results.html', message="Model already trained. You can now make predictions.")

    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect('/')

@app.route('/predict_future.html', methods=['GET', 'POST'])
def predict_future():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            input_features = [
                int(form_data['Year']),
                int(form_data['Month']),
                form_data['DayOfWeek'],
                int(form_data['Hour']),
                float(form_data['Latitude']),
                float(form_data['Longitude'])
            ]

            scaler = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb'))
            le_day = pickle.load(open(os.path.join(MODEL_PATH, 'le_day.pkl'), 'rb'))
            le_crime = pickle.load(open(os.path.join(MODEL_PATH, 'le_crime.pkl'), 'rb'))
            knn = pickle.load(open(os.path.join(MODEL_PATH, 'KNN.pkl'), 'rb'))

            input_features[2] = le_day.transform([input_features[2]])[0]
            input_features = scaler.transform([input_features])

            prediction = knn.predict(input_features)[0]
            crime_type = le_crime.inverse_transform([prediction])[0]

            flash(f'Predicted Future Crime: {crime_type}')
            return render_template('predict_future.html', prediction=crime_type)

        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect(request.url)

    return render_template('predict_future.html')

@app.route('/visualize.html')
def visualize():
    try:
        images = [f for f in os.listdir(STATIC_PATH) if f.startswith('confusion_matrix')]
        if not images:
            flash("No visualizations found. Train the model first!")
            return redirect('/train')
        return render_template('visualize.html', images=images)
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
