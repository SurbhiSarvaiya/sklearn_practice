from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # You can replace with SVM, Tree, etc.

# 1. Load data
iris = load_iris()
X = iris.data         # Features
y = iris.target       # Labels

# 2. Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)
print(predictions)
