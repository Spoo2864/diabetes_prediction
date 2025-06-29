from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('diabetes_prediction/svm_model.sav')
scaler = joblib.load('diabetes_prediction/scaler.save')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = [float(x) for x in request.form.values()]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
            return render_template('predict.html', prediction_text=result)

        except Exception as e:
            return render_template('predict.html', prediction_text=f"❌ Error: {str(e)}")

    return render_template('predict.html', prediction_text='')

if __name__ == '__main__':
    app.run(debug=True)
