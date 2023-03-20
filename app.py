from flask import Flask, request, jsonify
import numpy as np
import pickle
import sklearn

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hellooo!!"

@app.route('/predict', methods=['POST'])
def predict():
    Age = request.form.get('Age')
    Gender = request.form.get('Gender')
    HSC_P = request.form.get('HSC_P')
    SSC_P = request.form.get('SSC_P')
    #Stream = request.form.get('Stream')
    Internships = request.form.get('Internships')
    CGPA = request.form.get('CGPA')
    HistoryOfBacklogs = request.form.get('HistoryOfBacklogs')

    input_query = np.array([[Age, HSC_P, SSC_P, Gender, Internships, CGPA, HistoryOfBacklogs]])

    result = model.predict(input_query)[0]

    return jsonify({'Placed': str(int(result*100))})

if __name__ == '__main__':
    app.run(debug=True)
