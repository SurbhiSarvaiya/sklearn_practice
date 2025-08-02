from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model to file
joblib.dump(model, 'decision_tree_model.pkl')
print("Model saved successfully!")
# Load saved model
loaded_model = joblib.load('decision_tree_model.pkl')

# Use it
predictions = loaded_model.predict(X_test)
print("Predictions:", predictions)
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
