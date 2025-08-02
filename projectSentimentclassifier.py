# Sample data (you can expand this or use a CSV)
texts = [
    "I love this product!", "This is the best phone I've used.",
    "Horrible experience", "I hate this", 
    "Not bad, could be better", "Absolutely fantastic!",
    "Terrible service", "Very disappointing"
]

labels = [1, 1, 0, 0, 0, 1, 0, 0]  # 1 = positive, 0 = negative
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
# Predict new text
new_text = ["I really enjoyed the service"]
new_text_vec = vectorizer.transform(new_text)
prediction = model.predict(new_text_vec)
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")

import joblib

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load later
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


model = joblib.load('vectorizer.pkl')
print(model)