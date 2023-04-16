import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/IMDB Dataset.csv")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.sentiment.values

    # Stratified kfolds on input dataframe and storing it with "kfold" column
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    df.to_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/IMDB_folds.csv", index=False)