from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
data = joblib.load('model/wine_cultivar_model.pkl')
model = data['model']
scaler = data['scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    # Scale input
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    output = f"Cultivar {prediction[0] + 1}"
    return render_template('index.html', prediction_text=f'Predicted Wine Origin: {output}')

if __name__ == "__main__":
    app.run(debug=True)