import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load Model
MODEL_PATH = "naiveB.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML & CSS UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Analytics & Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #a855f7;
            --accent-pink: #ec4899;
            --accent-blue: #3b82f6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #c084fc, #f472b6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        input, select {
            width: 100%;
            padding: 0.85rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25);
        }

        select option {
            background-color: #0f172a;
            color: #fff;
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px -5px rgba(236, 72, 153, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(236, 72, 153, 0.6);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            animation: fadeIn 0.4s ease-in-out;
        }

        .result-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .result-value.class-0 {
            color: #4ade80;
        }

        .result-value.class-1 {
            color: #f87171;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Employee Retention Predictor</h1>
            <p>Powered by Naive Bayes Machine Learning</p>
        </div>

        <form method="POST" action="/predict" class="grid-form">
            <div class="form-group">
                <label>Education</label>
                <select name="Education" required>
                    <option value="0">Bachelors</option>
                    <option value="1">Masters</option>
                    <option value="2">PHD</option>
                </select>
            </div>

            <div class="form-group">
                <label>Joining Year</label>
                <input type="number" name="JoiningYear" min="2010" max="2030" value="2018" required>
            </div>

            <div class="form-group">
                <label>City</label>
                <select name="City" required>
                    <option value="0">Bangalore</option>
                    <option value="1">Pune</option>
                    <option value="2">New Delhi</option>
                </select>
            </div>

            <div class="form-group">
                <label>Payment Tier</label>
                <select name="PaymentTier" required>
                    <option value="1">Tier 1 (High)</option>
                    <option value="2">Tier 2 (Medium)</option>
                    <option value="3">Tier 3 (Standard)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Age</label>
                <input type="number" name="Age" min="18" max="70" value="28" required>
            </div>

            <div class="form-group">
                <label>Gender</label>
                <select name="Gender" required>
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label>Ever Benched</label>
                <select name="EverBenched" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label>Experience in Current Domain (Years)</label>
                <input type="number" name="ExperienceInCurrentDomain" min="0" max="20" value="3" required>
            </div>

            <button type="submit" class="btn-submit">Generate Prediction</button>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            <div class="result-title">Prediction Result</div>
            <div class="result-value class-{{ prediction_class }}">{{ prediction_text }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: Model file 'naiveB.pkl' not found.", prediction_class="1")

    try:
        # Extract features in exact order of feature_names_in_
        features = [
            float(request.form["Education"]),
            float(request.form["JoiningYear"]),
            float(request.form["City"]),
            float(request.form["PaymentTier"]),
            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["EverBenched"]),
            float(request.form["ExperienceInCurrentDomain"])
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        # Result mapping
        result_text = "Employee Likely to Leave (Leave Risk High)" if prediction == 1 else "Employee Likely to Stay (Retained)"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=result_text, prediction_class=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error processing input: {str(e)}", prediction_class="1")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
