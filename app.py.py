from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and encoders
model = joblib.load("OneDrive/Desktop/ONLINE_COURSE_RECOMMANDATION/course_recommender_model.pkl")
difficulty_encoder = joblib.load("OneDrive/Desktop/ONLINE_COURSE_RECOMMANDATION/difficulty_encoder.pkl")
scaler = joblib.load("OneDrive/Desktop/ONLINE_COURSE_RECOMMANDATION/difficulty_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        course_price = float(request.form['course_price'])
        course_duration = float(request.form['course_duration_hours'])
        feedback_score = float(request.form['feedback_score'])
        previous_courses = int(request.form['previous_courses_taken'])
        time_spent = float(request.form['time_spent_hours'])
        enrollment_numbers = int(request.form['enrollment_numbers'])

        # Scale input features
        input_data = np.array([[course_duration, course_price, feedback_score, 
                                time_spent, previous_courses, enrollment_numbers]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        label = difficulty_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction_text=f"Predicted Difficulty Level: {label}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True