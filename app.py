# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "best_aqi_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Load dataset to extract features
df = pd.read_csv(r"/Users/satyammukkawar/Downloads/AQI_dataset.csv")
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)

# Features
target_col = "AQI"
X = df.drop(columns=[target_col])
feature_names = X.columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        user_input = {}
        for feature in feature_names:
            value = request.form.get(feature)
            try:
                user_input[feature] = float(value) if value else 0.0
            except:
                user_input[feature] = 0.0

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Ensure missing columns exist
        for col in feature_names:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[feature_names]

        # Predict AQI
        try:
            predicted_aqi = model.predict(input_df)[0]
            predicted_aqi = round(float(predicted_aqi), 2)
        except Exception as e:
            return render_template('index.html', 
                                   features=feature_names,
                                   error=f"Prediction failed: {str(e)}")

        return render_template('result.html', 
                               prediction=predicted_aqi,
                               input=user_input)

    return render_template('index.html', features=feature_names)


if __name__ == '__main__':
    app.run(debug=True) 