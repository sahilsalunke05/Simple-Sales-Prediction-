from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("sales_model.pkl")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]

    return render_template('predict.html',
                           prediction=prediction,
                           input_data=features)

if __name__ == "__main__":
    app.run(debug=True)