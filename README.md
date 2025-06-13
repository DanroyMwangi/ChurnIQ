# ChurnIQ - Hybrid ML Architecture Demonstration

ChurnIQ is an educational hobby project under the MIT License that demonstrates **hybrid machine learning architecture** by combining traditional ML models with modern generative AI. This REST API showcases how classical statistical methods (scikit-learn) can work synergistically with cutting-edge AI (Google Gemini) to create more intelligent customer churn prediction systems for retail environments.

## 🎯 Overview

ChurnIQ demonstrates the power of **hybrid machine learning architecture** by combining traditional ML models with modern AI capabilities. This REST API showcases how classical machine learning (scikit-learn) can be enhanced with cutting-edge generative AI (Google Gemini) to create a more intelligent and actionable customer churn prediction system.

The project illustrates how traditional statistical models can provide the foundational predictions, while modern AI adds contextual understanding, business insights, and personalized strategies - creating a synergistic approach that's more powerful than either technology alone.

## ✨ Features

- **Hybrid ML Architecture**: Demonstrates the synergy between traditional ML and modern AI
- **REST API Design**: Pure JSON-based endpoints showcasing clean API architecture
- **Dual-Model Approach**: Linear regression + Random forest for comprehensive risk assessment
- **AI-Enhanced Intelligence**: Google Gemini AI adds contextual business understanding
- **Real-world Application**: Practical retail customer churn prediction use case
- **Educational Value**: Perfect example of how to integrate classical ML with generative AI
- **Production-Ready**: CORS enabled, comprehensive error handling, and scalable design

## 🔧 How It Works - Hybrid ML Pipeline

ChurnIQ demonstrates **hybrid machine learning architecture** by seamlessly integrating traditional ML with modern AI:

### 1. Traditional ML Foundation

- **Linear Regression Model**: Provides quantitative risk scoring using proven statistical methods
- **Random Forest Classifier**: Delivers reliable binary classification with ensemble learning
- **Feature Engineering**: Classical data preprocessing and standardized scaling
- **Robust Predictions**: Time-tested algorithms ensure consistent, reliable base predictions

### 2. Modern AI Enhancement Layer

- **Google Gemini AI Integration**: Adds contextual understanding and business intelligence
- **Semantic Analysis**: Interprets numerical predictions in real-world business context
- **Dynamic Insights**: Generates personalized recommendations based on customer segments
- **Natural Language Output**: Converts technical predictions into actionable business strategies

### 3. Hybrid Intelligence

- **Best of Both Worlds**: Traditional ML provides accuracy, AI provides understanding
- **Layered Architecture**: Each technology operates in its strength area
- **Enhanced Output**: Combined system delivers both precise predictions and meaningful insights
- **Practical Application**: Demonstrates real-world implementation of hybrid ML systems

### 4. API-First Design

- **Clean Separation**: Traditional ML and AI components work independently but collaboratively
- **Modular Architecture**: Easy to extend, modify, or replace individual components
- **Production Ready**: Proper error handling, validation, and response formatting

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
- Flask-CORS
- scikit-learn
- pandas
- numpy
- joblib
- python-dotenv
- google-generativeai (optional, for AI insights)

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
   # Edit .env with your Google API key for AI insights
   # GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the API server**

   ```bash
   python flask_app.py
   ```

5. **Test the API**

   The API will be available at `http://localhost:5000`

   ```bash
   curl http://localhost:5000/
   ```

## 📡 API Documentation

### Base URL

```text
http://localhost:5000
```

### Authentication

No authentication required for this version.

### Content Type

All requests and responses use `application/json` content type.

### Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Error description"
}
```

HTTP Status Codes:

- `200`: Success
- `400`: Bad Request (validation errors, missing fields)
- `500`: Internal Server Error (prediction failures, model errors)

---

### 1. API Information

**GET** `/`

Get basic API information and available endpoints.

#### API Information Response

```json
{
  "name": "ChurnIQ API",
  "version": "1.0.0",
  "description": "Customer churn prediction REST API",
  "endpoints": {
    "/": "API information",
    "/predict": "POST - Single customer churn prediction",
    "/predict_group": "POST - Multiple customers churn prediction",
    "/get_retention_strategy": "POST - Get personalized retention strategy"
  }
}
```

---

### 2. Single Customer Prediction

**POST** `/predict`

Predict churn risk for a single customer with detailed analysis.

#### Single Customer Request Body

```json
{
  "AverageSpendPerVisitKsh": 150.0,
  "Profit": 45.0,
  "FrequencyOfPurchases": 8.0,
  "average_time_btn_visits": 25.0,
  "time_since_last_visit": 10.0,
  "customer_lifetime": 2.5
}
```

#### Single Customer Request Parameters

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `AverageSpendPerVisitKsh` | float | Customer's average spending per store visit (USD) | Yes |
| `Profit` | float | Profit generated from this customer | Yes |
| `FrequencyOfPurchases` | float | Number of purchases in analysis period | Yes |
| `average_time_btn_visits` | float | Average days between store visits | Yes |
| `time_since_last_visit` | float | Days since customer's last visit | Yes |
| `customer_lifetime` | float | Total relationship duration (years) | Yes |

#### Single Customer Response

```json
{
  "success": true,
  "data": {
    "churn_risk_score": 0.27384204055323624,
    "churn_prediction": "Customer will not Churn",
    "will_churn": false,
    "customer_lifetime_value": 3000.0,
    "recommendations": [
      "Maintain current customer service excellence."
    ],
    "smart_insights": [
      "High-value, frequent customer. Risk is low, but recent activity is slowing (10 days since last visit). Send a personalized email highlighting new products or a special offer to encourage another purchase. Implement a loyalty program or VIP tier to further reward and retain this valuable customer."
    ],
    "customer_data": {
      "AverageSpendPerVisitKsh": 150.0,
      "Profit": 45.0,
      "FrequencyOfPurchases": 8.0,
      "average_time_btn_visits": 25.0,
      "time_since_last_visit": 10.0
    }
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `churn_risk_score` | float | Continuous risk score from linear regression model |
| `churn_prediction` | string | Human-readable prediction result |
| `will_churn` | boolean | Binary churn prediction (true = will churn) |
| `customer_lifetime_value` | float | Calculated CLV based on input data |
| `recommendations` | array | List of actionable recommendations |
| `smart_insights` | array | AI-generated insights from Gemini (if configured) |
| `customer_data` | object | Echo of input customer data |

---

### 3. Group Customer Prediction

**POST** `/predict_group`

Predict churn risk for multiple customers with group analysis.

#### Request Body (JSON Array Format)

```json
{
  "customers": [
    {
      "VisitorID": "CUST001",
      "FrequencyOfPurchases": 8.0,
      "AverageSpendPerVisitKsh": 150.0,
      "Profit": 45.0,
      "average_time_btn_visits": 25.0,
      "time_since_last_visit": 10.0
    },
    {
      "VisitorID": "CUST002",
      "FrequencyOfPurchases": 3.0,
      "AverageSpendPerVisitKsh": 80.0,
      "Profit": -5.0,
      "average_time_btn_visits": 60.0,
      "time_since_last_visit": 90.0
    }
  ]
}
```

#### Request Body (CSV Format)

```json
{
  "csv_data": "VisitorID,FrequencyOfPurchases,AverageSpendPerVisitKsh,Profit,average_time_btn_visits,time_since_last_visit\nCUST001,8.0,150.0,45.0,25.0,10.0\nCUST002,3.0,80.0,-5.0,60.0,90.0"
}
```

#### Group Prediction Request Parameters

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `customers` | array | Array of customer objects (use this OR csv_data) | Conditional |
| `csv_data` | string | CSV formatted customer data (use this OR customers) | Conditional |

#### Customer Object Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `VisitorID` | string | Unique customer identifier | Yes |
| `FrequencyOfPurchases` | float | Number of purchases in analysis period | Yes |
| `AverageSpendPerVisitKsh` | float | Customer's average spending per store visit (USD) | Yes |
| `Profit` | float | Profit generated from this customer | Yes |
| `average_time_btn_visits` | float | Average days between store visits | Yes |
| `time_since_last_visit` | float | Days since customer's last visit | Yes |

#### Group Prediction Response

```json
{
  "success": true,
  "data": {
    "summary": {
      "total_customers": 2,
      "total_churn": 1,
      "total_retain": 1,
      "churn_rate": 50.0,
      "total_revenue_at_risk": 80.0
    },
    "customers": [
      {
        "customer_id": "CUST001",
        "churn_risk_score": 0.27384204055323624,
        "churn_prediction": "Customer will not Churn",
        "will_churn": false,
        "recommendations": ["Maintain current customer service excellence."],
        "smart_insights": ["High-value, frequent customer..."],
        "customer_data": {
          "AverageSpendPerVisitKsh": 150.0,
          "Profit": 45.0,
          "FrequencyOfPurchases": 8.0,
          "average_time_btn_visits": 25.0,
          "time_since_last_visit": 10.0
        }
      }
    ],
    "group_insights": "Portfolio shows mixed risk levels. Immediate action needed on high-risk segment to prevent revenue loss...",
    "trends_analysis": "Market position concerning due to 50% churn rate. Key trend: customer retention declining. Growth opportunity: focus on loyalty programs."
  }
}
```

#### Summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_customers` | integer | Total number of customers analyzed |
| `total_churn` | integer | Number of customers predicted to churn |
| `total_retain` | integer | Number of customers predicted to stay |
| `churn_rate` | float | Percentage of customers predicted to churn |
| `total_revenue_at_risk` | float | Sum of average spend for at-risk customers |

---

### 4. Personalized Retention Strategy

**POST** `/get_retention_strategy`

Generate a personalized retention strategy for a specific customer.

#### Retention Strategy Request Body

```json
{
  "spend": 150.0,
  "frequency": 8.0,
  "profit": 45.0,
  "last_visit": 10.0,
  "churn_risk": 0
}
```

#### Retention Strategy Request Parameters

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `spend` | float | Average spend per visit | Yes |
| `frequency` | float | Purchase frequency | Yes |
| `profit` | float | Customer profit | Yes |
| `last_visit` | float | Days since last visit | Yes |
| `churn_risk` | integer | Churn risk level (0 = low, 1 = high) | Yes |

#### Retention Strategy Response

```json
{
  "success": true,
  "strategy": "Week 1-2: Send personalized email with exclusive 15% discount on favorite product categories to encourage immediate purchase. Week 3-4: Follow up with loyalty program invitation offering double points for next three visits. Success metric: Track visit frequency increase and average basket size growth over 30-day period."
}
```

---

## 📊 Example Usage

### cURL Examples

#### Single Customer Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AverageSpendPerVisitKsh": 150.0,
    "Profit": 45.0,
    "FrequencyOfPurchases": 8.0,
    "average_time_btn_visits": 25.0,
    "time_since_last_visit": 10.0,
    "customer_lifetime": 2.5
  }'
```

#### Group Prediction

```bash
curl -X POST http://localhost:5000/predict_group \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "VisitorID": "CUST001",
        "FrequencyOfPurchases": 8.0,
        "AverageSpendPerVisitKsh": 150.0,
        "Profit": 45.0,
        "average_time_btn_visits": 25.0,
        "time_since_last_visit": 10.0
      }
    ]
  }'
```

### Python Example

```python
import requests
import json

# Single customer prediction
url = "http://localhost:5000/predict"
data = {
    "AverageSpendPerVisitKsh": 150.0,
    "Profit": 45.0,
    "FrequencyOfPurchases": 8.0,
    "average_time_btn_visits": 25.0,
    "time_since_last_visit": 10.0,
    "customer_lifetime": 2.5
}

response = requests.post(url, json=data)
result = response.json()

if result["success"]:
    print(f"Churn Prediction: {result['data']['churn_prediction']}")
    print(f"Risk Score: {result['data']['churn_risk_score']:.3f}")
    print(f"CLV: ${result['data']['customer_lifetime_value']:.2f}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript Example

```javascript
// Single customer prediction
const predictChurn = async (customerData) => {
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(customerData)
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Churn Prediction:', result.data.churn_prediction);
      console.log('Risk Score:', result.data.churn_risk_score);
      console.log('Recommendations:', result.data.recommendations);
    } else {
      console.error('Error:', result.error);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
};

// Example usage
const customer = {
  AverageSpendPerVisitKsh: 150.0,
  Profit: 45.0,
  FrequencyOfPurchases: 8.0,
  average_time_btn_visits: 25.0,
  time_since_last_visit: 10.0,
  customer_lifetime: 2.5
};

predictChurn(customer);
```

## 🗂️ Project Structure

```text
ChurnIQ/
├── flask_app.py                                    # Main Flask REST API application
├── requirements.txt                                # Python dependencies
├── .env                                           # Environment configuration (optional)
├── churn-prediction-multiple-features.joblib      # Linear regression model
├── rf_model.joblib                                # Random forest classifier
├── scaler-multiple-features.joblib               # Feature scaler
├── final_churn_data.csv                          # Processed training dataset
├── sample_customers.csv                          # Sample data for testing
├── retrain_models.py                             # Model retraining script
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

## 📊 Hybrid ML Architecture

ChurnIQ showcases **hybrid machine learning architecture** by combining traditional statistical models with modern generative AI:

### Traditional ML Core

1. **Linear Regression**: Provides continuous risk scoring using established statistical methods
2. **Random Forest**: Delivers reliable binary classification with ensemble learning techniques

### Modern AI Enhancement

1. **Google Gemini AI**: Adds contextual business intelligence and natural language insights
2. **Collaborative Processing**: Traditional models generate predictions, AI interprets and enhances them

### Why This Approach Works

- **Complementary Strengths**: Traditional ML excels at pattern recognition, AI excels at contextual understanding
- **Reliability + Intelligence**: Proven algorithms provide accuracy, AI adds business value
- **Practical Implementation**: Demonstrates real-world integration of classical and modern ML techniques
- **Educational Value**: Perfect example for learning how to combine different ML generations

This hybrid approach results in a system that's both mathematically robust and business-intelligent, showcasing the power of hybrid machine learning architecture in practical applications.

## 🤖 Traditional ML + AI Hybrid System

ChurnIQ demonstrates how traditional machine learning and modern AI can work together synergistically:

### Traditional ML Layer (Foundation)

- **Quantitative Analysis**: Linear regression and random forest provide numerical predictions
- **Pattern Recognition**: Classical algorithms identify statistical relationships in customer data
- **Reliable Base Predictions**: Time-tested models ensure consistent, accurate risk scoring

### AI Enhancement Layer (Intelligence)

- **Contextual Understanding**: Gemini AI interprets numerical predictions in business context
- **Natural Language Insights**: Converts technical predictions into actionable business strategies
- **Dynamic Recommendations**: Generates personalized retention strategies based on customer segments
- **Business Intelligence**: Provides portfolio-level insights and trend analysis

### Hybrid Benefits

- **Accuracy + Understanding**: Traditional ML ensures mathematical precision, AI adds business value
- **Scalable Intelligence**: Classical models handle computation, AI provides interpretation
- **Best of Both Worlds**: Proven statistical methods enhanced with modern language understanding
- **Educational Demonstration**: Shows practical implementation of multi-generational ML integration

## 🎓 Project Purpose & Learning Outcomes

This hobby project demonstrates key concepts in **hybrid machine learning architecture**:

### Educational Goals

- **Traditional ML Mastery**: Understanding linear regression, random forests, and ensemble methods
- **Modern AI Integration**: Learning to incorporate generative AI into classical ML workflows
- **API Design**: Building production-ready REST APIs with proper error handling and documentation
- **Hybrid Intelligence**: Showcasing how different ML generations can work together synergistically

### Technical Demonstrations

- **Data Pipeline**: From raw retail data to trained models and real-time predictions
- **Model Serving**: Deploying ML models via REST API with proper scaling and validation
- **AI Enhancement**: Using LLMs to add business context to technical predictions
- **Architecture Patterns**: Clean separation between traditional ML and AI components

### Practical Applications

- **Real-world Problem**: Customer churn prediction in retail environments
- **Business Value**: Converting technical predictions into actionable business strategies
- **Scalable Design**: Architecture that can handle both single predictions and batch processing
- **Production Readiness**: Comprehensive error handling, validation, and response formatting

This project serves as a comprehensive example of how modern data science combines traditional statistical methods with cutting-edge AI to create more intelligent and valuable systems.

## 🚀 Deployment

### Production Considerations

For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI instead of Flask's development server

   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
   ```

2. **Environment Variables**: Set up proper environment configuration

   ```bash
   export FLASK_ENV=production
   export GOOGLE_API_KEY=your_api_key_here
   ```

3. **Reverse Proxy**: Use nginx for load balancing and SSL termination

4. **Monitoring**: Implement logging and error tracking

5. **Database**: Consider storing predictions for analytics and model improvement

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app:app"]
```

Build and run:

```bash
docker build -t churniq-api .
docker run -p 5000:5000 -e GOOGLE_API_KEY=your_key churniq-api
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Add tests and improve code coverage

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests before submitting PRs

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- AI insights powered by Google Gemini
- Training data from [WisdomAxis Sample Superstore Dataset](https://www.wisdomaxis.com/technology/software/tableau/sample-data/)
- Original dataset designed for Tableau analytics and retail business intelligence

## 📞 Support

For questions, issues, or feature requests:

- Open an issue on GitHub
- Check the API documentation above
- Review the example usage patterns

---

**ChurnIQ** - Demonstrating Hybrid Machine Learning Architecture: Where Traditional ML Meets Modern AI
