from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the model and scaler files
model_path = os.path.join(BASE_DIR, 'churn-prediction-multiple-features.joblib')
scaler_path = os.path.join(BASE_DIR, 'scaler-multiple-features.joblib')
rf_model_path = os.path.join(BASE_DIR, 'rf_model.joblib')

# Load the saved models and scalers
model = load(model_path)
scaler = load(scaler_path)
rf_model = load(rf_model_path)

def generate_recommendation(row, churn_prediction):
    recommendations = []
    time_since_last_visit_threshold = 75
    avg_time_between_visits_threshold = 90
    spend_threshold = 90
    profit_threshold = -10
    frequency_threshold = 5
    
    if row['time_since_last_visit'] > time_since_last_visit_threshold:
        recommendations.append(f"Re-engage with personalized marketing campaigns. (Time since last store visit: {row['time_since_last_visit']} days)")

    if row['average_time_btn_visits'] > avg_time_between_visits_threshold:
        recommendations.append(f"Offer incentives for more frequent store visits. (Average time between visits: {row['average_time_btn_visits']} days)")

    if row['Profit'] < profit_threshold:
        recommendations.append(f"Review pricing strategy or product mix. (Current loss: ${row['Profit']})")
        
    if row['FrequencyOfPurchases'] < frequency_threshold:
        recommendations.append(f"Implement loyalty programs to encourage repeat shopping. (Current frequency: {row['FrequencyOfPurchases']})")
        
    if row['AverageSpendPerVisitKsh'] < spend_threshold:
        recommendations.append(f"Introduce promotions to increase basket size per visit. (Current average spend: ${row['AverageSpendPerVisitKsh']})")
    
    if churn_prediction == 1:  # 1 indicates churn
        if not recommendations:  # If list is empty
            recommendations.append("Engage with targeted offers and discounts.")
    else:
        if not recommendations:  # If no issues found
            recommendations.append("Maintain current customer service excellence.")

    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting and converting form data
    average_spend = float(request.form['AverageSpendPerVisitKsh'])
    profit = float(request.form['Profit'])
    frequency_of_purchases = float(request.form['FrequencyOfPurchases'])
    average_time_btn_visits = float(request.form['average_time_btn_visits'])
    time_since_last_visit = float(request.form['time_since_last_visit'])

    # Calculate Customer Lifetime Value (Assuming a lifespan of 3 years)
    customer_lifespan = float(request.form['customer_lifetime'])  # years
    clv = average_spend * frequency_of_purchases * customer_lifespan

    # Prepare features for model prediction and predict
    features = np.array([average_spend, profit, frequency_of_purchases, average_time_btn_visits, time_since_last_visit]).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Make predictions
    lin_reg_prediction = model.predict(features_scaled)[0]
    rf_prediction = rf_model.predict(features_scaled)[0]

    if rf_prediction == 1:
        randomForestPrediction = 'Customer will Churn'
    else:
        randomForestPrediction = 'Customer will not Churn'

    visitor_row = {
        'AverageSpendPerVisitKsh': average_spend,
        'Profit': profit,
        'FrequencyOfPurchases': frequency_of_purchases,
        'average_time_btn_visits': average_time_btn_visits,
        'time_since_last_visit': time_since_last_visit
    }

    recommendations = generate_recommendation(visitor_row, rf_prediction)

    return render_template('result.html', lin_reg_prediction=lin_reg_prediction, rf_prediction=randomForestPrediction, recommendations=recommendations, clv=clv)

@app.route('/predict_group', methods=['POST'])
def predict_group():
    from io import StringIO
    
    # Get CSV data from form
    csv_data = request.form['csv_data']
    
    # Parse CSV data
    try:
        df = pd.read_csv(StringIO(csv_data))
        
        # Validate required columns
        required_columns = ['VisitorID', 'FrequencyOfPurchases', 'AverageSpendPerVisitKsh', 'Profit', 'average_time_btn_visits', 'time_since_last_visit']
        if not all(col in df.columns for col in required_columns):
            return render_template('result.html', error="Missing required columns. Please ensure your CSV has: VisitorID, FrequencyOfPurchases, AverageSpendPerVisitKsh, Profit, average_time_btn_visits, time_since_last_visit")
        
        results = []
        total_churn = 0
        
        # Process each row
        for index, row in df.iterrows():
            customer_id = row['VisitorID']
            frequency_of_purchases = row['FrequencyOfPurchases']
            average_spend = row['AverageSpendPerVisitKsh']
            profit = row['Profit']
            average_time_btn_visits = row['average_time_btn_visits']
            time_since_last_visit = row['time_since_last_visit']
            
            # Prepare features for model prediction
            features = np.array([average_spend, profit, frequency_of_purchases, average_time_btn_visits, time_since_last_visit]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Make predictions
            lin_reg_prediction = model.predict(features_scaled)[0]
            rf_prediction = rf_model.predict(features_scaled)[0]
            
            will_churn = rf_prediction == 1
            if will_churn:
                total_churn += 1
                rf_prediction_text = 'Customer will Churn'
            else:
                rf_prediction_text = 'Customer will not Churn'
            
            visitor_row = {
                'AverageSpendPerVisitKsh': average_spend,
                'Profit': profit,
                'FrequencyOfPurchases': frequency_of_purchases,
                'average_time_btn_visits': average_time_btn_visits,
                'time_since_last_visit': time_since_last_visit
            }
            
            recommendations = generate_recommendation(visitor_row, rf_prediction)
            
            result = {
                'customerId': customer_id,
                'regressionPrediction': lin_reg_prediction,
                'randomForestPrediction': rf_prediction_text,
                'recommendations': recommendations,
                'willChurn': will_churn
            }
            results.append(result)
        
        return render_template('group_results.html', results=results, total_churn=total_churn, total_customers=len(results))
        
    except Exception as e:
        return render_template('result.html', error=f"Error processing CSV data: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)