from gensim.models import Word2Vec

sentences = [["i", "like", "nlp"], ["i", "enjoy", "machine", "learning"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

print(model.wv['nlp'])  # Vector for 'nlp'
print(model.wv.most_similar('learning'))  # Similar words
