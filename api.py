from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="IT Ticket Classifier API",
    description="API for classifying IT support tickets into categories",
    version="1.0.0"
)

# Load the trained model and vectorizer
try:
    classifier = joblib.load('it_ticket_classifier.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to generate the model files.")
    classifier = None
    vectorizer = None

class TicketRequest(BaseModel):
    description: str

class TicketResponse(BaseModel):
    category: str
    confidence: float

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "IT Ticket Classification API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Predict ticket category",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if classifier is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=TicketResponse)
async def predict_category(ticket: TicketRequest):
    """
    Predict the category of an IT support ticket
    
    Args:
        ticket: TicketRequest object containing the ticket description
        
    Returns:
        TicketResponse object with predicted category and confidence score
    """
    if classifier is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Vectorize the input text
        vectorized_text = vectorizer.transform([ticket.description])
        
        # Get prediction and probability
        category = classifier.predict(vectorized_text)[0]
        probabilities = classifier.predict_proba(vectorized_text)[0]
        confidence = max(probabilities)
        
        return TicketResponse(
            category=category,
            confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing ticket: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
