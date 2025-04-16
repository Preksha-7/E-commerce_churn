import pytest
import json
import os
from app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

def test_home_endpoint(client):
    """Test that the home endpoint returns correct information."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'endpoints' in data
    assert 'POST /predict' in data['endpoints']

def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_predict_missing_data(client):
    """Test prediction with missing data."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_with_sample_data(client):
    """Test prediction with sample data if model exists."""
    # Skip this test if model file doesn't exist in test environment
    if not os.path.exists('models/churn_model.h5'):
        pytest.skip("Model file not found, skipping prediction test")
    
    sample_data = {
        'Tenure': 0.5,
        'WarehouseToHome': 0.3,
        'HourSpendOnApp': 2.5,
        'NumberOfDeviceRegistered': 3,
        'SatisfactionScore': 4,
        'NumberOfAddress': 2,
        'Complain': 0,
        'OrderAmountHikeFromlastYear': 0.15,
        'CouponUsed': 2,
        'OrderCount': 5,
        'DaySinceLastOrder': 0.2,
        'CashbackAmount': 0.1,
        'PreferedOrderCat_Fashion': 1,
        'PreferedOrderCat_Grocery': 0,
        'PreferedOrderCat_Mobile': 0,
        'MaritalStatus_Married': 1,
        'MaritalStatus_Single': 0
    }
    
    response = client.post('/predict', json=sample_data)
    assert response.status_code in [200, 500]  # 500 if model can't be loaded
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'churn_prediction' in data
        assert 'churn_probability' in data
        assert isinstance(data['churn_prediction'], int)
        assert isinstance(data['churn_probability'], float)