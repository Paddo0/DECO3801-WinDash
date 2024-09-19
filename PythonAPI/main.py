from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
#from firebase_config import db
from RNN import get_pred
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Path to the service account key 
#cred = credentials.Certificate('path-service-account')

# Initialization of the app 
#firebase_admin.initialize_app(cred)

# Initialize Firestore DB
#db = firestore.client()

@app.route('/add', methods=['POST'])
def post_prediction():
    # TODO - Implement a method to post next prediction/s to the app
    try:
        data = request.get_json()
        number = data['number']
        return jsonify({"message": f"Received Prediction: {number}"})
    except Exception as e:
        return f"An Error Occurred: {e}", 400

@app.route('/prediction', methods=['GET'])
def get_prediction():
    try:
        prediction = get_pred()  # 调用 RNN 中的预测函数
        print(f"Prediction from get_pred: {prediction}")
        return jsonify({"prediction": prediction})  # 返回预测结果
    except Exception as e:
        return f"An Error Occurred: {e}", 400


@app.route('/data', methods=['GET'])
def get_data():
    return "Database integration pending"

"""
@app.route('/pred', methods=['GET'])
def get_data():
    # TODO - Implement a method to retrieve real-time data from the database 
    try:
        prediction = get_pred()  # Call the get_pred function from RNN
        print(f"Prediction from get_pred: {prediction}")
        return jsonify({"prediction": prediction})  # Return the prediction result
    except Exception as e:
        return f"An Error Occurred: {e}", 400
"""
    

def main():
    # TODO - Implement a python API using Flask or any other API tool to return machine learning predictions
    print("TODO")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
