import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt') # Need to run this line of code only the first time


sentence = "hi, how are you?"

print("NORMAL SPLIT : ", sentence.split())

print("Tokenisation", word_tokenize(sentence))