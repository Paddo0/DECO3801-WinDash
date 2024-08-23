
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
#from firebase_config import db
from RNN import get_pred



app = Flask(__name__)

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

@app.route('/pred', methods=['GET'])
def get_data():
    # TODO - Implement a method to retreive the real time data from the database 
    prediction = get_pred()

    try:
         pass
    except Exception as e:
            return f"An Error Occurred: {e}", 400
    
    return f"\nFirst prediction is {prediction}"

def main():
    # TODO - Implement a python api using flask or any other api tool to return machine learning predictions
    print("TODO")

     


if __name__ == "__main__":
    main()
