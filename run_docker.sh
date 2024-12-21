#!/bin/bash
docker build -t ecommerce-churn-api .
docker run -p 5000:5000 ecommerce-churn-api
