# IT Ticket Classification System

This project implements a machine learning system for classifying IT support tickets into different categories. It includes both the model training pipeline and a FastAPI deployment for real-time predictions.

## Features

- Automatic classification of IT tickets into 5 categories:
  - Network Issues
  - Software Installation
  - Hardware Problems
  - Password Reset
  - Email Issues
- REST API endpoint for real-time predictions
- Model performance evaluation with accuracy and F1-score metrics
- Sample data generation for testing and demonstration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```
This will:
- Generate sample training data
- Train the classifier
- Save the model and vectorizer
- Generate a performance report

3. Start the API server:
```bash
python api.py
```
The API will be available at http://localhost:8000

## API Usage

### Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `POST /predict`: Predict ticket category

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"description": "Cannot connect to internet"}'
```

Example Response:
```json
{
    "category": "Network Issues",
    "confidence": 0.92
}
```

## Project Structure

- `requirements.txt`: Project dependencies
- `train_model.py`: Script for generating data and training the model
- `api.py`: FastAPI implementation for model deployment
- `it_ticket_classifier.joblib`: Trained model file (generated after training)
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer (generated after training)
- `model_performance_report.txt`: Model evaluation metrics and insights

## Model Details

The system uses:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
- Multinomial Naive Bayes classifier for prediction
- Scikit-learn for model implementation

## Performance

Check `model_performance_report.txt` after training for detailed metrics including:
- Accuracy
- F1-score
- Per-category performance metrics
