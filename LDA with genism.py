from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Example documents
docs = ["I like to eat broccoli and bananas",
        "I ate a banana and spinach smoothie",
        "Chinchillas and kittens are cute"]

# Preprocess
stop_words = set(stopwords.words('english'))
texts = [[word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words]
         for doc in docs]

# Dictionary and Corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA Model
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print topics
topics = lda_model.print_topics()
for topic in topics:
    print(topic)
