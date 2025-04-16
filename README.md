# E-Commerce Customer Churn Prediction

A machine learning solution to predict customer churn for e-commerce businesses using deep learning and MLOps pipelines.

## Project Overview

This project builds a predictive model that helps e-commerce businesses identify customers who are likely to churn (stop purchasing). By predicting churn probability, businesses can take proactive measures to retain valuable customers.

### Features

- **Data Preprocessing Pipeline**: Handles missing values, feature encoding, and scaling
- **Machine Learning Models**: TensorFlow neural network model for churn prediction
- **REST API**: Flask-based API for real-time predictions
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Docker Containerization**: Containerized application for easy deployment

## Project Structure

```
E-commerce_churn/
├── .github/
│   └── workflows/          # GitHub Actions workflows
├── data/                   # Data directory (gitignored)
│   └── raw_data_placeholder.txt
├── models/                 # Model directory (gitignored)
├── scripts/                # Utility scripts
│   ├── retrain_model.py    # Script for model retraining
│   └── run_pipeline.py     # End-to-end pipeline script
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── train_test_split.py # Data splitting and balancing
│   ├── model_building.py   # Model training
│   ├── evaluate_model.py   # Model evaluation
│   └── serialize.py        # Model serialization
├── tests/                  # Test files
│   └── test_app.py         # API tests
├── .gitignore              # Git ignore file
├── app.py                  # Flask API
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── run_docker.sh           # Docker run script
```

## Installation and Setup

### Prerequisites

- Python 3.9+
- Docker (for containerization)
- Git

### Local Setup

1. Clone the repository

   ```
   git clone https://github.com/yourusername/E-commerce_churn.git
   cd E-commerce_churn
   ```

2. Create a virtual environment

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```
   pip install -r requirements.txt
   ```

4. Place your dataset in the `data/` directory

   ```
   cp your_dataset.csv data/data_ecommerce_customer_churn.csv
   ```

5. Run the data processing and model training pipeline
   ```
   python scripts/run_pipeline.py
   ```

### Docker Setup

1. Build and run the Docker container

   ```
   bash run_docker.sh
   ```

   Or manually:

   ```
   docker build -t ecommerce-churn-api .
   docker run -p 5000:5000 ecommerce-churn-api
   ```

## API Usage

### Endpoints

- **GET /** - Home endpoint with API information
- **GET /health** - Health check endpoint
- **POST /predict** - Prediction endpoint

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 0.5,
    "WarehouseToHome": 0.3,
    "HourSpendOnApp": 2.5,
    "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 4,
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 0.15,
    "CouponUsed": 2,
    "OrderCount": 5,
    "DaySinceLastOrder": 0.2,
    "CashbackAmount": 0.1,
    "PreferedOrderCat_Fashion": 1,
    "PreferedOrderCat_Grocery": 0,
    "PreferedOrderCat_Mobile": 0,
    "MaritalStatus_Married": 1,
    "MaritalStatus_Single": 0
}'
```

### Example Response

```json
{
  "status": "success",
  "churn_prediction": 0,
  "churn_probability": 0.23,
  "message": "Customer likely to stay"
}
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. On each push to the main branch, the workflow:
   - Checks out the code
   - Sets up Python
   - Installs dependencies
   - Runs tests
   - Builds and pushes the Docker image

## Model Retraining

To retrain the model with new data:

1. Place new data in `data/latest_customer_data.csv`
2. Run the retraining script:
   ```
   python scripts/retrain_model.py
   ```

## License

[MIT License](LICENSE)
