# E-Commerce Customer Churn Prediction

A comprehensive machine learning solution to predict customer churn for e-commerce businesses using deep learning and advanced data science techniques.

## Project Overview

This project builds a predictive model that helps e-commerce businesses identify customers who are likely to churn (stop purchasing). By predicting churn probability, businesses can take proactive measures to retain valuable customers and optimize their marketing strategies.

### Key Features

- **Advanced Data Preprocessing**: Comprehensive pipeline handling missing values, feature engineering, and data normalization
- **Deep Learning Model**: TensorFlow neural network with dropout layers for robust churn prediction
- **Class Imbalance Handling**: SMOTE implementation to address dataset imbalance
- **Interactive Web Interface**: Clean, responsive HTML interface for real-time predictions
- **REST API**: Flask-based API for seamless integration with existing systems
- **Model Persistence**: Multiple serialization formats for model deployment flexibility

## Project Structure

```
E-commerce_churn/
├── data/                   # Data directory
│   └── raw_data_placeholder.txt
├── models/                 # Trained models storage
├── scripts/                # Utility scripts
│   ├── retrain_model.py    # Model retraining pipeline
│   └── run_pipeline.py     # End-to-end pipeline execution
├── src/                    # Core source code
│   ├── preprocessing.py    # Data preprocessing and feature engineering
│   ├── train_test_split.py # Data splitting with SMOTE balancing
│   ├── model_building.py   # Neural network architecture and training
│   ├── evaluate_model.py   # Model performance evaluation
│   └── serialize.py        # Model serialization utilities
├── static/                 # Web interface assets
│   └── index.html          # Interactive prediction interface
├── tests/                  # Test suite
│   └── test_app.py         # API endpoint testing
├── app.py                  # Flask API application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation and Setup

### Prerequisites

- Python 3.9+
- Git

### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/E-commerce_churn.git
   cd E-commerce_churn
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**

   ```bash
   # Place your dataset in the data directory
   cp your_dataset.csv data/data_ecommerce_customer_churn.csv
   ```

5. **Run the complete ML pipeline**

   ```bash
   python scripts/run_pipeline.py
   ```

6. **Start the API server**

   ```bash
   python app.py
   ```

## Data Requirements

The model expects the following features in your dataset:

- **Tenure**: Customer relationship duration (normalized 0-1)
- **WarehouseToHome**: Distance from warehouse to customer (normalized 0-1)
- **HourSpendOnApp**: Time spent on mobile application
- **NumberOfDeviceRegistered**: Number of devices registered per customer
- **SatisfactionScore**: Customer satisfaction rating (1-5)
- **NumberOfAddress**: Number of addresses associated with customer
- **Complain**: Binary indicator for customer complaints
- **OrderAmountHikeFromlastYear**: Percentage increase in order amount
- **CouponUsed**: Number of coupons utilized
- **OrderCount**: Total number of orders placed
- **DaySinceLastOrder**: Days since last purchase (normalized 0-1)
- **CashbackAmount**: Cashback amount received (normalized 0-1)
- **PreferedOrderCat**: Preferred order category (Fashion/Grocery/Mobile)
- **MaritalStatus**: Customer marital status (Married/Single)
- **Churn**: Target variable (1 = churned, 0 = retained)

## Model Architecture

The neural network architecture includes:

```python
Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

**Key Technical Decisions:**
- **Optimizer**: Adam with binary crossentropy loss
- **Regularization**: Dropout layers to prevent overfitting
- **Activation**: ReLU for hidden layers, sigmoid for output
- **Training**: 20 epochs with validation monitoring

## API Usage

### Available Endpoints

- `GET /` - API information and feature requirements
- `GET /health` - Health check endpoint
- `GET /sample` - Sample data for testing
- `POST /predict` - Churn prediction endpoint
- `GET /ui` - Interactive web interface

### Prediction Request Example

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

### Response Format

```json
{
  "status": "success",
  "churn_prediction": 0,
  "churn_probability": 0.23,
  "message": "Customer likely to stay"
}
```

## Web Interface

Access the interactive prediction interface at `http://localhost:5000/ui` after starting the server. The interface provides:

- **User-friendly form** with input validation
- **Real-time predictions** with probability visualization
- **Responsive design** for desktop and mobile
- **Probability bar** showing churn likelihood

## Model Performance

The model achieves:
- **Balanced accuracy** through SMOTE oversampling
- **Robust predictions** with dropout regularization
- **Comprehensive evaluation** using classification reports
- **Probability calibration** for business decision-making

## Model Retraining

Update your model with new data:

1. **Place new data** in `data/latest_customer_data.csv`
2. **Run retraining script**:
   ```bash
   python scripts/retrain_model.py
   ```
3. **Restart the API** to load the updated model

## Testing

Run the test suite:

```bash
pytest tests/test_app.py -v
```

Tests cover:
- API endpoint functionality
- Data validation
- Model loading and prediction
- Error handling scenarios

## Technical Stack

- **Backend**: Flask, TensorFlow, Scikit-learn
- **Data Processing**: Pandas, NumPy, Imbalanced-learn
- **Frontend**: HTML, CSS, JavaScript
- **Model Persistence**: HDF5, Pickle
- **Testing**: Pytest

## Business Impact

This solution enables businesses to:
- **Identify at-risk customers** before they churn
- **Optimize retention strategies** based on churn probability
- **Reduce customer acquisition costs** through proactive retention
- **Improve customer lifetime value** through targeted interventions

## License

[MIT License](LICENSE)

## Contact

**Preksha Upadhyay**  
Email: prekshaupadhyay03@gmail.com
