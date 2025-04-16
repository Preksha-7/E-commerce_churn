#!/usr/bin/env python
"""
E-commerce Customer Churn Prediction Pipeline

This script runs the end-to-end pipeline for data preprocessing,
model training, evaluation, and serialization.
"""
import os
import sys
import time
import logging

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        'data', 'data/raw', 'data/processed', 
        'models', 'logs'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Create directories first, before setting up logging
ensure_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ChurnPipeline")

def run_step(step_name, script_path):
    """Run a pipeline step and log its execution"""
    logger.info(f"Starting step: {step_name}")
    start_time = time.time()
    
    # Run the script
    exit_code = os.system(f"python {script_path}")
    
    if exit_code != 0:
        logger.error(f"Step '{step_name}' failed with exit code {exit_code}")
        return False
    
    duration = time.time() - start_time
    logger.info(f"Completed step: {step_name} in {duration:.2f} seconds")
    return True

def main():
    """Run the full data pipeline"""
    # Define pipeline steps
    pipeline_steps = [
        ("Data Preprocessing", "src/preprocessing.py"),
        ("Train-Test Split", "src/train_test_split.py"),
        ("Model Training", "src/model_building.py"),
        ("Model Evaluation", "src/evaluate_model.py"),
        ("Model Serialization", "src/serialize.py")
    ]
    
    # Run each step
    for step_name, script_path in pipeline_steps:
        success = run_step(step_name, script_path)
        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            return False
    
    logger.info("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    logger.info("Starting E-commerce Churn Prediction Pipeline")
    result = main()
    if result:
        logger.info("Pipeline execution completed successfully")
    else:
        logger.error("Pipeline execution failed")
        sys.exit(1)