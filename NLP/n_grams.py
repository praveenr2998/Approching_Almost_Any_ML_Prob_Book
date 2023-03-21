from nltk import ngrams
from nltk.tokenize import word_tokenize

N = 2

sentence = 'Hi, how are you?'
tokenized_sentence = word_tokenize(sentence)

n_grams = list(ngrams(tokenized_sentence, N))
print(n_grams)