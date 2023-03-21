from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

corpus = [
    "hello, how are you?",
    "You know football is a wonderful sport, what do you think?",
    "Opensource is something that everyone should appreciate, what do you think?"
]
tfidf = TfidfVectorizer()
tfidf.fit(corpus)

print("Unique index assigned to each word : ", tfidf.vocabulary_)

# TFIDF vectorizer
corpus_transformed = tfidf.transform(corpus)
print("Sparse Matrix Representation : ", corpus_transformed)


# To also include special characters while creating sparse matrix
tfidf = TfidfVectorizer(tokenizer = word_tokenize, token_pattern=None)
tfidf.fit(corpus)

print("Unique index assigned to each word : ", tfidf.vocabulary_)

# TFIDF vectorizer 
corpus_transformed = tfidf.transform(corpus)
print("Sparse Matrix Representation : ", corpus_transformed)



# SPARSE MATRIX(ASSUMPTION)
# Y axis ---> Sentence Number
# X axis ---> Unique word's index created by count vectorizer
# Values filled in Matrix ---> Count of each unique word in a sentence created by count vectorizer  