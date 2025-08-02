import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
# Step 1:Dataset
#You can use the popular SMS Spam Collection Dataset (also works for emails).
#📥 Download: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
#The dataset has two columns:
# Step 2: Load and Explore the Data
# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']
# Preview
print(df.head())
print(df['label'].value_counts())
#🧹 Step 3: Preprocessing
# Convert labels to 0 (ham) and 1 (spam)
df['label'] = LabelEncoder().fit_transform(df['label'])  # ham → 0, spam → 1
# Optional: text preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text
df['message'] = df['message'].apply(clean_text)
#📊 Step 4: Train-Test Split
from sklearn.model_selection import train_test_split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#📚 Step 5: Convert Text to Features (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#🤖 Step 6: Train the Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Predict
y_pred = model.predict(X_test_tfidf)
# Evaluate
print(classification_report(y_test, y_pred))
#💾 Step 7: Save the Model
import joblib
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
#🚀 Step 8: Make Predictions on New Emails
# Load model
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
# Sample message
new_message = ["Congratulations! You've won a $1000 gift card."]
cleaned = clean_text(new_message[0])
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)
print("Spam" if prediction[0] == 1 else "Not Spam")
