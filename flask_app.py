from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from joblib import load
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

# Initialize Gemini client
try:
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
except Exception as e:
    print(f"Warning: Gemini API not configured properly: {e}")
    client = None

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

def generate_smart_insights(customer_data, churn_prediction, recommendations):
    """
    Generate smart insights using Gemini AI based on customer data and predictions
    """
    if not client:
        return ["Smart insights unavailable - Gemini API not configured"]
    
    try:
        # Calculate additional metrics for better insights
        visit_frequency = 365 / customer_data['average_time_btn_visits'] if customer_data['average_time_btn_visits'] > 0 else 0
        revenue_per_transaction = customer_data['AverageSpendPerVisitKsh']
        profit_margin = (customer_data['Profit'] / customer_data['AverageSpendPerVisitKsh'] * 100) if customer_data['AverageSpendPerVisitKsh'] > 0 else 0
        
        # Prepare the enhanced prompt for Gemini
        prompt = f"""
        Analyze this customer and provide brief, actionable insights. No markdown formatting.
        
        Customer: ${customer_data['AverageSpendPerVisitKsh']:.0f} avg spend, {customer_data['FrequencyOfPurchases']} purchases, {customer_data['time_since_last_visit']} days since last visit. 
        Risk: {"HIGH" if churn_prediction == 1 else "LOW"}
        
        Provide 3-4 short bullet points with:
        - Customer type (high-value, frequent, etc)
        - Main risk factors if any
        - 2 specific actions to take
        
        Keep response under 200 words, no formatting, just simple bullet points.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for more consistent insights
                max_output_tokens=150,
                top_p=0.8,
                top_k=40
            )
        )
        
        return [response.text]
        
    except Exception as e:
        print(f"Error generating smart insights: {e}")
        return [f"Smart insights temporarily unavailable. Error: {str(e)}"]

def generate_group_insights(group_results, total_churn, total_customers):
    """
    Generate smart insights for group analysis using Gemini AI
    """
    if not client:
        return "Group insights unavailable - Gemini API not configured"
    
    try:
        # Calculate advanced metrics
        churn_rate = (total_churn / total_customers) * 100
        high_risk_customers = [r for r in group_results if r['willChurn']]
        
        # Calculate aggregate statistics
        avg_spend = sum(r.get('AverageSpendPerVisitKsh', 0) for r in group_results if 'AverageSpendPerVisitKsh' in str(r)) / total_customers
        total_revenue_at_risk = sum(r.get('AverageSpendPerVisitKsh', 0) for r in high_risk_customers)
        
        # Sample customer data for pattern analysis
        sample_data = group_results[:8] if len(group_results) > 8 else group_results
        
        prompt = f"""
        Analyze customer portfolio. No markdown. Simple insights only.
        
        Portfolio: {total_customers} customers, {churn_rate:.0f}% churn rate, ${total_revenue_at_risk:.0f} revenue at risk.
        
        Provide brief analysis:
        - Overall risk assessment (high/medium/low)
        - Top 2 immediate actions needed
        - Expected impact if no action taken
        
        Keep under 150 words, no formatting.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=120,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating group insights: {e}")
        return f"Group insights temporarily unavailable. Error: {str(e)}"

def generate_business_trends_analysis(group_results):
    """
    Generate business trends and predictive insights using Gemini AI
    """
    if not client or len(group_results) == 0:
        return "Trends analysis unavailable"
    
    try:
        # Calculate trend metrics
        high_risk_count = sum(1 for r in group_results if r['willChurn'])
        avg_frequency = np.mean([r.get('FrequencyOfPurchases', 0) for r in group_results])
        avg_spend = np.mean([r.get('AverageSpendPerVisitKsh', 0) for r in group_results])
        
        prompt = f"""
        Business trends analysis. Keep simple.
        
        Customer base: {len(group_results)} customers, {high_risk_count} high risk, average spend ${avg_spend:.0f}.
        
        Brief insights:
        - Market position (good/concerning/poor)
        - Key trend to watch
        - One growth opportunity
        
        Under 100 words, no formatting.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=80,
                top_p=0.9
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating trends analysis: {e}")
        return f"Trends analysis unavailable: {str(e)}"

def generate_personalized_retention_strategy(customer_data, churn_prediction):
    """
    Generate personalized retention strategy using Gemini AI
    """
    if not client:
        return "Personalized strategy unavailable - Gemini API not configured"
    
    try:
        # Determine customer segment
        spend_per_visit = customer_data['AverageSpendPerVisitKsh']
        frequency = customer_data['FrequencyOfPurchases']
        profit_margin = (customer_data['Profit'] / spend_per_visit * 100) if spend_per_visit > 0 else 0
        
        segment = "High-Value" if spend_per_visit > 100 and frequency > 10 else \
                 "Frequent" if frequency > 15 else \
                 "Occasional" if frequency > 5 else "Infrequent"
        
        prompt = f"""
        Create simple retention plan. No formatting.
        
        Customer: {segment} customer, ${spend_per_visit:.0f} spend, {frequency} purchases, {customer_data['time_since_last_visit']} days since visit.
        Risk: {"HIGH" if churn_prediction == 1 else "LOW"}
        
        Provide:
        - Week 1-2: One specific action
        - Week 3-4: One follow-up action
        - Success metric to track
        
        Under 120 words, plain text only.
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=100,
                top_p=0.85
            )
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating retention strategy: {e}")
        return f"Retention strategy unavailable: {str(e)}"

@app.route('/')
def home():
    """
    API Info endpoint
    """
    return jsonify({
        'name': 'ChurnIQ API',
        'version': '1.0.0',
        'description': 'Customer churn prediction REST API',
        'endpoints': {
            '/': 'API information',
            '/predict': 'POST - Single customer churn prediction',
            '/predict_group': 'POST - Multiple customers churn prediction',
            '/get_retention_strategy': 'POST - Get personalized retention strategy'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single customer churn prediction
    Expected JSON payload:
    {
        "AverageSpendPerVisitKsh": float,
        "Profit": float,
        "FrequencyOfPurchases": float,
        "average_time_btn_visits": float,
        "time_since_last_visit": float,
        "customer_lifetime": float
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['AverageSpendPerVisitKsh', 'Profit', 'FrequencyOfPurchases', 
                          'average_time_btn_visits', 'time_since_last_visit', 'customer_lifetime']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Extract and convert data
        average_spend = float(data['AverageSpendPerVisitKsh'])
        profit = float(data['Profit'])
        frequency_of_purchases = float(data['FrequencyOfPurchases'])
        average_time_btn_visits = float(data['average_time_btn_visits'])
        time_since_last_visit = float(data['time_since_last_visit'])
        customer_lifespan = float(data['customer_lifetime'])

        # Calculate Customer Lifetime Value
        clv = average_spend * frequency_of_purchases * customer_lifespan

        # Prepare features for model prediction
        features = np.array([average_spend, profit, frequency_of_purchases, average_time_btn_visits, time_since_last_visit]).reshape(1, -1)
        
        # Create feature names to match what the scaler expects
        feature_names = ['AverageSpendPerVisitKsh', 'Profit', 'FrequencyOfPurchases', 'average_time_btn_visits', 'time_since_last_visit']
        features_df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(features_df)

        # Make predictions
        lin_reg_prediction = model.predict(features_scaled)[0]
        rf_prediction = rf_model.predict(features_scaled)[0]

        will_churn = bool(rf_prediction == 1)
        rf_prediction_text = 'Customer will Churn' if will_churn else 'Customer will not Churn'

        visitor_row = {
            'AverageSpendPerVisitKsh': average_spend,
            'Profit': profit,
            'FrequencyOfPurchases': frequency_of_purchases,
            'average_time_btn_visits': average_time_btn_visits,
            'time_since_last_visit': time_since_last_visit
        }

        recommendations = generate_recommendation(visitor_row, rf_prediction)
        smart_insights = generate_smart_insights(visitor_row, rf_prediction, recommendations)

        return jsonify({
            'success': True,
            'data': {
                'churn_risk_score': float(lin_reg_prediction),
                'churn_prediction': rf_prediction_text,
                'will_churn': will_churn,
                'customer_lifetime_value': float(clv),
                'recommendations': recommendations,
                'smart_insights': smart_insights,
                'customer_data': visitor_row
            }
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid data format: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict_group', methods=['POST'])
def predict_group():
    """
    Group customer churn prediction
    Expected JSON payload:
    {
        "customers": [
            {
                "VisitorID": string,
                "FrequencyOfPurchases": float,
                "AverageSpendPerVisitKsh": float,
                "Profit": float,
                "average_time_btn_visits": float,
                "time_since_last_visit": float
            }
        ]
    }
    OR
    {
        "csv_data": "CSV string format"
    }
    """
    try:
        from io import StringIO
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        df = None
        
        # Handle CSV data format
        if 'csv_data' in data:
            try:
                df = pd.read_csv(StringIO(data['csv_data']))
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid CSV format: {str(e)}'
                }), 400
        
        # Handle JSON array format
        elif 'customers' in data:
            try:
                df = pd.DataFrame(data['customers'])
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid customers data format: {str(e)}'
                }), 400
        
        else:
            return jsonify({
                'success': False,
                'error': 'Either "csv_data" or "customers" field is required'
            }), 400
        
        # Validate required columns
        required_columns = ['VisitorID', 'FrequencyOfPurchases', 'AverageSpendPerVisitKsh', 
                           'Profit', 'average_time_btn_visits', 'time_since_last_visit']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing_columns}'
            }), 400
        
        results = []
        total_churn = 0
        
        # Process each row
        for index, row in df.iterrows():
            try:
                customer_id = row['VisitorID']
                frequency_of_purchases = float(row['FrequencyOfPurchases'])
                average_spend = float(row['AverageSpendPerVisitKsh'])
                profit = float(row['Profit'])
                average_time_btn_visits = float(row['average_time_btn_visits'])
                time_since_last_visit = float(row['time_since_last_visit'])
                
                # Prepare features for model prediction
                features = np.array([average_spend, profit, frequency_of_purchases, average_time_btn_visits, time_since_last_visit]).reshape(1, -1)
                
                # Create feature names to match what the scaler expects
                feature_names = ['AverageSpendPerVisitKsh', 'Profit', 'FrequencyOfPurchases', 'average_time_btn_visits', 'time_since_last_visit']
                features_df = pd.DataFrame(features, columns=feature_names)
                features_scaled = scaler.transform(features_df)
                
                # Make predictions
                lin_reg_prediction = model.predict(features_scaled)[0]
                rf_prediction = rf_model.predict(features_scaled)[0]
                
                will_churn = bool(rf_prediction == 1)
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
                smart_insights = generate_smart_insights(visitor_row, rf_prediction, recommendations)
                
                result = {
                    'customer_id': customer_id,
                    'churn_risk_score': float(lin_reg_prediction),
                    'churn_prediction': rf_prediction_text,
                    'will_churn': will_churn,
                    'recommendations': recommendations,
                    'smart_insights': smart_insights,
                    'customer_data': visitor_row
                }
                results.append(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error processing customer {row.get("VisitorID", index)}: {str(e)}'
                }), 400
        
        # Generate group insights using Gemini AI
        group_insights = generate_group_insights(results, total_churn, len(results))
        trends_analysis = generate_business_trends_analysis(results)
        
        # Calculate summary statistics
        churn_rate = (total_churn / len(results)) * 100 if len(results) > 0 else 0
        high_risk_customers = [r for r in results if r['will_churn']]
        total_revenue_at_risk = sum(r['customer_data']['AverageSpendPerVisitKsh'] for r in high_risk_customers)
        
        return jsonify({
            'success': True,
            'data': {
                'summary': {
                    'total_customers': len(results),
                    'total_churn': total_churn,
                    'total_retain': len(results) - total_churn,
                    'churn_rate': round(churn_rate, 2),
                    'total_revenue_at_risk': round(total_revenue_at_risk, 2)
                },
                'customers': results,
                'group_insights': group_insights,
                'trends_analysis': trends_analysis
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Group prediction failed: {str(e)}'
        }), 500

@app.route('/get_retention_strategy', methods=['POST'])
def get_retention_strategy():
    """
    Generate personalized retention strategy for a specific customer
    """
    try:
        data = request.get_json()
        customer_data = {
            'AverageSpendPerVisitKsh': float(data['spend']),
            'FrequencyOfPurchases': float(data['frequency']),
            'Profit': float(data['profit']),
            'time_since_last_visit': float(data['last_visit'])
        }
        churn_prediction = int(data['churn_risk'])
        
        strategy = generate_personalized_retention_strategy(customer_data, churn_prediction)
        
        return jsonify({
            'success': True,
            'strategy': strategy
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)