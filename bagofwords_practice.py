from sklearn.feature_extraction.text import CountVectorizer

docs = ["I like NLP", "I like machine learning"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())