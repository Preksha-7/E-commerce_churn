# Use the official Python image
FROM python:3.9-slim

# Set the working directory to /E-commerce_churn
WORKDIR /E-commerce_churn

# Copy all files from the current directory (on host) into the container
COPY . /E-commerce_churn

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port (change if not 5000)
EXPOSE 5000

# Set the default command to run your Python script
CMD ["python", "app.py"]
