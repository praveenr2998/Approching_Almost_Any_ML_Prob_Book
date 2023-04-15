import numpy as np
import io
import pandas as pd
import fasttext

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer



def load_vector_generating_model(model):
    model = fasttext.load_model(model)
    return model


def sentence_to_vec(s, vector_model, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    M = []
    for w in words:
        M.append(vector_model.get_word_vector(w))

    if len(M) == 0:
        return np.zeros(300)

    M = np.array(M)
    v = M.sum(axis=0)
    return v/np.sqrt((v**2).sum()) 


if __name__ == '__main__':
    df = pd.read_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/IMDB Dataset.csv")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
    df = df.sample(frac=1).reset_index(drop=True)
    vector_generating_model = load_vector_generating_model('/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/models/imdb_embedding.bin')
    print("Embeddings are loaded")

    vectors = []
    for review in df.review.values:
        vectors.append(sentence_to_vec(s = review, vector_model=vector_generating_model, stop_words=[], tokenizer=word_tokenize))

    vectors = np.array(vectors)
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracy_list = []
    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training Fold : {fold_}")
        xtrain = vectors[t_, :]
        ytrain = y[t_]

        xtest = vectors[v_, :]
        ytest = y[v_]

        model = linear_model.LogisticRegression()
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        accuracy = metrics.accuracy_score(ytest, preds)
        accuracy_list.append(accuracy)
        print(f"Accuracy Score : {accuracy}")
        print("")

    for i in range(0, 4):
        print(f"Fold : {i+1}, Accuracy : {accuracy_list[i]}") 



