from gensim.models import KeyedVectors

# Load pretrained GloVe (converted to Word2Vec format)
glove_model = KeyedVectors.load_word2vec_format("glove.6B.100d.word2vec.txt", binary=False)

print(glove_model['king'])  # Vector for 'king'
print(glove_model.most_similar('king'))
