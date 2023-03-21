from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

corpus = [
    "hello, how are you?",
    "You know football is a wonderful sport, what do you think?",
    "Opensource is something that everyone should appreciate, what do you think?"
]
ctv = CountVectorizer()
ctv.fit(corpus)

print("Unique index assigned to each word : ", ctv.vocabulary_)

# Count vectorizer - Coverts each word to a unique key and per sentence it displays the key along with count
corpus_transformed = ctv.transform(corpus)
print("Sparse Matrix Representation : ", corpus_transformed)


# To also include special characters while creating sparse matrix
ctv = CountVectorizer(tokenizer = word_tokenize, token_pattern=None)
ctv.fit(corpus)

print("Unique index assigned to each word : ", ctv.vocabulary_)

# Count vectorizer - Coverts each word to a unique key and per sentence it displays the key along with count
corpus_transformed = ctv.transform(corpus)
print("Sparse Matrix Representation : ", corpus_transformed)



# SPARSE MATRIX(ASSUMPTION)
# Y axis ---> Sentence Number
# X axis ---> Unique word's index created by count vectorizer
# Values filled in Matrix ---> Count of each unique word in a sentence created by count vectorizer  