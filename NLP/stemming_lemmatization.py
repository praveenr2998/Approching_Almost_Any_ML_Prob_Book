from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')

words = ['fishing', 'reading', 'corpora']

for word in words:
    print(f"Word ==> {word}")
    print(f"Stemmed Word ==> {stemmer.stem(word)}")
    print(f"Lemma ==> {lemmatizer.lemmatize(word)}")
    print(" ")