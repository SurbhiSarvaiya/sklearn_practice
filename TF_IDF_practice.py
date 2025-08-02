from sklearn.feature_extraction.text import TfidfVectorizer
docs = ["I like NLP", "I like machine learning"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())