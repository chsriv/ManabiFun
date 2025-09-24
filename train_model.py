import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json

def create_training_data():
    """Create synthetic training data for weakness detection"""
    np.random.seed(42)
    
    # Features: [grammar_score, articles_score, synonyms_score, antonyms_score, sentence_score, time_spent]
    # Labels: 0=grammar, 1=articles, 2=synonyms, 3=antonyms, 4=sentences
    
    data = []
    labels = []
    
    # Generate 1000 synthetic student profiles
    for i in range(1000):
        # Simulate different weakness patterns
        weakness = np.random.randint(0, 5)
        
        # Base scores (0-100)
        scores = np.random.normal(75, 15, 5)  # Average 75% with std 15
        scores = np.clip(scores, 0, 100)
        
        # Make the weakness area significantly lower
        scores[weakness] = np.random.normal(45, 10)  # Weak area: 45% average
        scores[weakness] = np.clip(scores[weakness], 0, 80)
        
        # Time spent (minutes) - more time on weak areas
        time_spent = np.random.normal(20, 5)
        if weakness == 0:  # grammar weakness
            time_spent += np.random.normal(10, 3)
        
        feature_row = list(scores) + [time_spent]
        data.append(feature_row)
        labels.append(weakness)
    
    return np.array(data), np.array(labels)

def train_weakness_detector():
    """Train the ML model to detect student weaknesses"""
    print("🤖 Creating training data...")
    X, y = create_training_data()
    
    print("🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("📊 Training model...")
    # Use Random Forest for interpretability
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model accuracy: {accuracy:.2%}")
    
    # Save model
    with open('models/weakness_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("💾 Model saved to models/weakness_detector.pkl")
    return model

def create_questions_dataset():
    """Create the questions database"""
    topics = {
        "grammar": "Grammar Basics Island 🏝️",
        "articles": "Articles Island 🏖️", 
        "synonyms": "Synonyms Island 🌴",
        "antonyms": "Antonyms Island 🏛️",
        "sentences": "Sentence Formation Island 🏰"
    }
    
    # Sample questions (you'll expand this to 50 per topic)
    questions_db = {
        "grammar": [
            {
                "question": "Which is correct?",
                "options": ["I are happy", "I am happy", "I is happy", "I be happy"],
                "answer": 1,
                "difficulty": "easy"
            },
            {
                "question": "Choose the plural form of 'child':",
                "options": ["childs", "children", "childes", "child"],
                "answer": 1,
                "difficulty": "medium"
            },
            # Add 48 more grammar questions...
        ],
        "articles": [
            {
                "question": "Fill in the blank: '___ apple is red'",
                "options": ["A", "An", "The", "No article"],
                "answer": 1,
                "difficulty": "easy"
            },
            # Add 49 more articles questions...
        ],
        "synonyms": [
            {
                "question": "What is a synonym for 'happy'?",
                "options": ["sad", "angry", "joyful", "tired"],
                "answer": 2,
                "difficulty": "easy"
            },
            # Add 49 more synonyms questions...
        ],
        "antonyms": [
            {
                "question": "What is the opposite of 'hot'?",
                "options": ["warm", "cold", "cool", "mild"],
                "answer": 1,
                "difficulty": "easy"
            },
            # Add 49 more antonyms questions...
        ],
        "sentences": [
            {
                "question": "Which sentence is correct?",
                "options": ["The dog run fast", "The dog runs fast", "The dog running fast", "The dog ran fastly"],
                "answer": 1,
                "difficulty": "medium"
            },
            # Add 49 more sentence questions...
        ]
    }
    
    # Save questions database
    with open('data/questions.json', 'w') as f:
        json.dump(questions_db, f, indent=2)
    
    print("📚 Questions database created!")
    return questions_db

if __name__ == "__main__":
    import os
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("🚀 Starting ManabiFun ML Training...")
    
    # Create questions dataset
    create_questions_dataset()
    
    # Train weakness detection model
    train_weakness_detector()
    
    print("✅ Setup complete! Ready to run the Streamlit app.")
