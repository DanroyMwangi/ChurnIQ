# ChurnIQ

ChurnIQ is an open-source hobby project under the MIT License that uses traditional machine-learning models trained on publicly available supermarket/retail data to predict customer churn and generate actionable recommendations for customer retention in retail environments.

## 🎯 Overview

ChurnIQ combines the power of classical machine learning with intelligent recommendation systems to help retail businesses understand and prevent customer churn. The application analyzes customer shopping behavior patterns and provides both individual and group predictions with personalized retention strategies specifically designed for supermarket chains and retail environments.

## ✨ Features

- **Single Customer Prediction**: Analyze individual customer churn risk with detailed recommendations
- **Group Analysis**: Process multiple customers simultaneously using CSV data input
- **Interactive Web Interface**: Clean, modern UI with tabbed navigation
- **Actionable Insights**: Receive specific recommendations based on customer behavior patterns
- **Real-time Analysis**: Instant predictions with risk scoring and retention strategies

## 🔧 How It Works

1. **Data Input**  
   - Single customer: Fill out the web form with customer metrics
   - Multiple customers: Paste CSV data directly into the text area
   
2. **Machine Learning Analysis**  
   - **Linear Regression Model**: Provides risk scoring for churn probability
   - **Random Forest Classifier**: Binary classification (will churn / won't churn)
   - **Feature Engineering**: Analyzes shopping patterns, visit frequency, and retail behavioral metrics
   
3. **Recommendation Engine**  
   - Evaluates customer behavior against predefined thresholds
   - Generates personalized retention strategies
   - Provides actionable insights for customer engagement
   
4. **Results Dashboard**  
   - Individual results with recommendations and Customer Lifetime Value (CLV)
   - Group results with summary statistics and expandable recommendations
   - Visual indicators for churn risk levels

## 📊 Input Features

The model analyzes the following customer shopping metrics:

- **Average Spend Per Visit** (USD): Customer's typical transaction amount per store visit
- **Profit**: Revenue generated from the customer after costs
- **Frequency of Purchases**: Number of shopping transactions in the analysis period
- **Average Time Between Visits**: Days between customer store visits
- **Time Since Last Visit**: Days since the customer's most recent shopping trip
- **Customer Lifetime**: Total relationship duration with the retail chain (for CLV calculation)

## 📈 Dataset & Training Data

ChurnIQ is trained on the **Sample Superstore** dataset, a popular retail analytics dataset that contains comprehensive information about customer transactions, product sales, and shopping behavior patterns.

### Data Source
- **Primary Source**: [WisdomAxis Sample Data](https://www.wisdomaxis.com/technology/software/tableau/sample-data/)
- **Dataset**: Sample Superstore data for retail analytics
- **Backup Files**: Local copies included in the repository for data persistence
  - `Sample - Superstore for Tableau 9.x versions.xls`
  - `Sample - Superstore Sales (Excel) for Tableau 8.x versions.xls`
  - `final_churn_data.csv` (processed dataset used for model training)

### Data Characteristics
- **Industry**: Retail/Supermarket chain
- **Scope**: Customer shopping behavior, purchase patterns, and transaction history
- **Features**: Customer demographics, product categories, sales performance, and temporal patterns
- **Privacy**: All data is anonymized and publicly available for educational/research purposes

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ChurnIQ.git
   cd ChurnIQ
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

4. **Run the application**
   ```bash
   python flask_app.py
   ```

5. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## 📈 Usage

### Single Customer Prediction

1. Navigate to the "Single Prediction" tab
2. Fill in the customer metrics form
3. Click "Predict" to get instant results
4. Review the churn prediction, risk score, and personalized recommendations

### Group Analysis

1. Switch to the "Group Prediction" tab
2. Prepare your CSV data with the required columns:
   ```
   VisitorID,FrequencyOfPurchases,AverageSpendPerVisitKsh,Profit,average_time_btn_visits,time_since_last_visit
   ```
3. Paste the CSV data into the text area
4. Click "Predict Group" to analyze all customers
5. Review the summary statistics and individual customer results

## 🗂️ Project Structure

```text
ChurnIQ/
├── flask_app.py                                    # Main Flask application
├── requirements.txt                                # Python dependencies
├── .env                                           # Environment configuration
├── templates/                                     # HTML templates
│   ├── index.html                                # Main interface with tabs
│   ├── result.html                               # Single prediction results
│   └── group_results.html                        # Group prediction results
├── churn-prediction-multiple-features.joblib      # Linear regression model
├── rf_model.joblib                                # Random forest classifier
├── scaler-multiple-features.joblib               # Feature scaler
├── final_churn_data.csv                          # Processed training dataset
├── Sample - Superstore for Tableau 9.x versions.xls    # Original Tableau data (9.x)
├── Sample - Superstore Sales (Excel) for Tableau 8.x versions.xls  # Original Tableau data (8.x)
├── LICENSE                                        # MIT License
└── README.md                                      # This file
```

## 🎨 Recommendation Types

ChurnIQ provides targeted recommendations based on retail customer behavior:

- **Re-engagement**: For customers with long absence periods from the store
- **Visit Frequency**: Incentives for customers with irregular shopping patterns  
- **Pricing Strategy**: Review for customers with negative profit margins
- **Loyalty Programs**: Encourage repeat purchases for low-frequency shoppers
- **Spending Promotions**: Boost average transaction values and basket size
- **General Retention**: Maintain service quality for low-risk customers

## 📊 Models & Performance

The application uses two complementary models trained on supermarket retail data:

1. **Linear Regression**: Provides continuous risk scoring for churn probability
2. **Random Forest**: Binary churn classification with high accuracy

Both models are trained on the Sample Superstore dataset and optimized for retail customer behavior analysis, specifically focusing on shopping patterns, purchase frequency, and customer lifetime value in supermarket environments.

## 🔮 Future Enhancements

- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Real-time model retraining capabilities
- [ ] Enhanced visualization dashboards
- [ ] API endpoints for programmatic access
- [ ] Integration with CRM systems
- [ ] A/B testing framework for recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- UI styled with Tailwind CSS
- Training data from [WisdomAxis Sample Superstore Dataset](https://www.wisdomaxis.com/technology/software/tableau/sample-data/)
- Original dataset designed for Tableau analytics and retail business intelligence
