import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Generate sample IT ticket data
def generate_sample_data(n_samples=1000):
    categories = [
        'Network Issues',
        'Software Installation',
        'Hardware Problems',
        'Password Reset',
        'Email Issues'
    ]
    
    # Sample descriptions for each category
    descriptions = {
        'Network Issues': [
            "Cannot connect to internet",
            "Wifi is slow",
            "Network drive not accessible",
            "VPN connection failed",
            "Unable to access network printer"
        ],
        'Software Installation': [
            "Need Microsoft Office installed",
            "Software update required",
            "Application not working after update",
            "New software installation request",
            "Program crashes on startup"
        ],
        'Hardware Problems': [
            "Computer not turning on",
            "Screen display issues",
            "Keyboard not working",
            "Mouse is unresponsive",
            "Printer not working"
        ],
        'Password Reset': [
            "Forgot password",
            "Need to reset account password",
            "Password expired",
            "Cannot login to system",
            "Account locked out"
        ],
        'Email Issues': [
            "Cannot send emails",
            "Email not syncing",
            "Missing emails",
            "Email attachment problems",
            "Outlook not responding"
        ]
    }
    
    tickets = []
    labels = []
    
    for _ in range(n_samples):
        category = np.random.choice(categories)
        description = np.random.choice(descriptions[category])
        # Add some random variations to make descriptions more unique
        if np.random.random() > 0.5:
            description += f" - User ID: {np.random.randint(1000, 9999)}"
        tickets.append(description)
        labels.append(category)
    
    return pd.DataFrame({'description': tickets, 'category': labels})

# Train the model
def train_classifier():
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['description'], 
        df['category'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create and fit the vectorizer
    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train the classifier
    print("Training the classifier...")
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(classifier, 'it_ticket_classifier.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    # Generate performance report
    performance_report = f"""
Model Performance Report:
------------------------
The IT ticket classification model achieved {accuracy:.2%} accuracy and {f1:.2%} F1-score on the test set. 
The model can effectively categorize tickets into 5 categories: Network Issues, Software Installation, 
Hardware Problems, Password Reset, and Email Issues. The model uses TF-IDF vectorization and 
Multinomial Naive Bayes classification.

Detailed Classification Report:
{report}
    """
    
    # Save the report
    with open('model_performance_report.txt', 'w') as f:
        f.write(performance_report)
    
    print("Training completed successfully!")
    return performance_report

if __name__ == "__main__":
    train_classifier()
