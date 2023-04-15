import fasttext
import pandas as pd
import re

df = pd.read_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/IMDB Dataset.csv")


def preprocess(text):
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = re.sub(r'[ \n]+', ' ', text)
    return text.strip().lower()

df.review = df.review.map(preprocess)

df.to_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/training_data.txt", columns=['review'], header=None, index=False)

model = fasttext.train_unsupervised('/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/training_data.txt')
model.save_model("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/models/imdb_embedding.bin")



