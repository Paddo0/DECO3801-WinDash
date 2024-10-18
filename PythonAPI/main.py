from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
from predict import test

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# initialize Firestore
cred = credentials.Certificate('deco-windash-firebase-adminsdk-u0tv1-f05fb8cc6f.json')  # Replace with your service account path

firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route('/add', methods=['POST'])
def post_prediction():
    # TODO - Implement a method to post next prediction(s) to the app
    try:
        data = request.get_json()
        number = data['number']
        return jsonify({"message": f"Received Prediction: {number}"})
    except Exception as e:
        return f"An Error Occurred: {e}", 400


@app.route('/daily-prediction', methods=['POST'])          
def get_daily_prediction():
    try:
        # Receive meterId
        data = request.get_json()
        meter_id = data.get('meterId')

        print(f"Received daily prediction request for meter ID: {meter_id}")  

        # Get document form dailyData in firebase
        doc_ref = db.collection('dailyData').document(meter_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"No document found for meter ID: {meter_id}")  
            return jsonify({"error": "No data found for the given meter ID"}), 404

        meter_data = doc.to_dict()
        print(f"Document data for meter ID {meter_id}: {meter_data}")  

        # Get Intensity and pass to the model
        intensity_values = [entry['Intensity'] for entry in meter_data['seriesData']]
        print(f"Intensity values for prediction: {intensity_values}")

        # Use the model to pred
        prediction = test(intensity_values)
        print(f'predictions is {prediction}')

        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  
        return jsonify({"error": str(e)}), 500


@app.route('/monthly-prediction', methods=['POST'])
def get_monthly_prediction():
    try:
        # Get meterId
        data = request.get_json()
        meter_id = data.get('meterId')

        # Get month statistics from firebase
        doc_ref = db.collection('overallData').document(meter_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": "No data found for the given meter ID"}), 404

        meter_data = doc.to_dict()
        
        # Get `AverageIntensity` from data
        intensity_values = [entry['AverageIntensity'] for entry in meter_data['data']]
        print(f"Intensity values for prediction: {intensity_values}")

        # Use the model to pred
        
        prediction = test(intensity_values)
        
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/data', methods=['GET'])
def get_data():
    return "Database integration pending"




def main():
    print("TODO")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)  # the host is local laptop, be careful with that
