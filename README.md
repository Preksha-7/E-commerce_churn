# E-commerce Customer Churn Prediction System

## Overview

This project implements a deep learning model using TensorFlow to predict customer churn for e-commerce businesses. The system analyzes historical data, transaction patterns, and customer interactions to identify potential churners, enabling proactive retention strategies and optimized customer engagement.

## Business Value

- **Reduce Revenue Loss**: Identify at-risk customers before they churn
- **Optimize Retention Budget**: Focus resources on customers with highest churn probability
- **Improve Customer Experience**: Address pain points identified through model insights
- **Increase ROI**: Target retention efforts where they'll have the greatest impact

## Architecture

The project follows a modern MLOps architecture with the following components:

1. **Data Pipeline**: Preprocessing of customer data including handling missing values, encoding categorical features, and feature scaling
2. **Model Training**: Deep learning model built with TensorFlow using sequential layers with dropout for regularization
3. **Model Evaluation**: Comprehensive evaluation using accuracy, precision, recall, and F1-score
4. **API Service**: Flask-based REST API for real-time predictions
5. **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
6. **Docker Container**: Containerized application for consistent deployment across environments

## Technologies Used

- **Python 3.9+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model development
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and metrics
- **Flask**: Web framework for API development
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline
- **SMOTE**: Handling class imbalance

## Project Structure

```
├── .github/workflows/     # CI/CD configuration
├── data/                  # Data storage (gitignored)
├── models/                # Trained models (gitignored)
├── scripts/               # Utility scripts
│   ├── retrain_model.py   # Script for model retraining
│   └── run_pipeline.py    # End-to-end pipeline runner
├── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing
│   ├── train_test_split.py# Data splitting and SMOTE balancing
│   ├── model_building.py  # Model architecture and training
│   ├── evaluate_model.py  # Model evaluation
│   └── serialize.py       # Model serialization
├── tests/                 # Unit and integration tests
├── .gitignore
├── Dockerfile             # Docker configuration
├── README.md              # Project documentation
├── app.py                 # Flask API application
├── requirements.txt       # Python dependencies
└── run_docker.sh          # Docker run script
```

## Features

- **Deep Neural Network**: Multi-layer neural network with dropout layers to prevent overfitting
- **Advanced Preprocessing**: Handling of missing values, categorical encoding, and feature scaling
- **Class Imbalance Handling**: SMOTE oversampling to address class imbalance in churn prediction
- **REST API Interface**: Flask API for real-time predictions
- **Containerized Deployment**: Easy deployment via Docker
- **CI/CD Integration**: Automated testing and deployment through GitHub Actions

## Model Architecture

The core prediction model is a sequential deep neural network with:

- Input layer matching the feature dimensions
- Hidden layers with 128 and 64 neurons using ReLU activation
- Dropout layers (0.3 and 0.2) for regularization
- Output layer with sigmoid activation for binary classification

## Installation and Setup

### Prerequisites

- Python 3.9+
- Docker (optional)

### Local Setup

1. Clone the repository:

   ```
   git clone https://github.com/your-username/ecommerce-churn-prediction.git
   cd ecommerce-churn-prediction
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the complete pipeline:

   ```
   python scripts/run_pipeline.py
   ```

4. Start the API server:
   ```
   python app.py
   ```

### Docker Setup

1. Build and run with Docker:
   ```
   ./run_docker.sh
   ```
   Or manually:
   ```
   docker build -t ecommerce-churn-api .
   docker run -p 5000:5000 ecommerce-churn-api
   ```

## API Usage

The API provides a simple interface for churn predictions:

### Predict Endpoint

**POST /predict**

Example request:

```json
{
  "Tenure": 10,
  "WarehouseToHome": 15,
  "NumberOfDeviceRegistered": 3,
  "SatisfactionScore": 4,
  "DaySinceLastOrder": 7,
  "CashbackAmount": 120,
  "PreferedOrderCat_Fashion": 1,
  "PreferedOrderCat_Grocery": 0,
  "PreferedOrderCat_Mobile": 0,
  "MaritalStatus_Married": 1,
  "MaritalStatus_Single": 0
}
```

Example response:

```json
{
  "churn_prediction": 0,
  "churn_probability": 0.12
}
```

## Development

### Adding New Features

1. Develop new feature code in the appropriate module
2. Add tests in the `tests/` directory
3. Run tests using `pytest tests/`
4. Submit a pull request

### Retraining the Model

To retrain the model with new data:

1. Place new data in `data/latest_customer_data.csv`
2. Run `python scripts/retrain_model.py`

## Continuous Integration/Deployment

The project uses GitHub Actions for CI/CD:

- Automated testing on push to main branch
- Docker image building and publishing to Docker Hub

## Future Improvements

- Implement model explainability using SHAP or LIME
- Add feature importance visualization
- Develop customer segmentation for targeted retention strategies
- Implement A/B testing framework for retention campaigns
- Create dashboard for monitoring model performance

## Contact

[prekshaupadhyay03@gmail.com] Preksha Upadhyay
