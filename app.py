from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('nbclassifier (1).pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data and convert to float
        int_features = [float(x) for x in request.form.values()]
        input_data = np.array([int_features])
        print(input_data)
        input_data_scaled = scaler.transform(input_data)
        print(input_data_scaled)
        prediction = model.predict(input_data_scaled)
        
        output = 'True' if prediction[0] == 1 else 'False'
    except Exception as e:
        output = f'Error: {str(e)}'
    print(output)
    return render_template('index.html', prediction_text=f'This user has Anaemia: {output}')

if __name__ == '__main__':
    app.run(debug=True)
