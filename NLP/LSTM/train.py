import io
import torch
import numpy as np
import pandas as pd
import tensorrt
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import fasttext
from sklearn import metrics

import config
import dataset
import engine
import lstm

def load_vectors(model_path):
    '''
    This functions is used to load the pretrained fasttext model which is used to vectorize text
    
    INPUT
    model_path : String - path where model is stored
    
    RETURNS
    fasttext model
    '''
    vector_generating_model = fasttext.load_model(model_path)
    return vector_generating_model


def create_embedding_matrix(word_index, vector_generating_model):
    '''
    This function is used to create the embedding matrix from the tokenizer's word_index, it uses the fasttext model to get vectors

    INPUT
    word_index              : dictionary - word as key, index as values
    vector_generating_model : fasttext pretrained model

    RETURNS
    embedding_matrix : index as key, vector of the corresponding word as value
    '''
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_matrix[i] = vector_generating_model.get_word_vector(word)
    return embedding_matrix


def run(df, fold):
    '''
    This function is the orchestrator which executes the training/validation process step by step

    INPUT
    df   : pandas dataframe - Input
    fold : int - fold number(kfold)

    RETURNS
    None
    '''
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = tf.keras.preprocessing.text.Tokenizer() # Using keras tokenizer
    tokenizer.fit_on_texts(df.review.values.tolist())

    xtrain = tokenizer.texts_to_sequences(train_df.review.values) # Each letter will be converted, eg 'Good day' ===> [[1],[2],[2],[3],[],[3],[4],[5]
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN) # Padding from the left with zeros, if length exceeds max length --> truncate from left  
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)

    train_dataset = dataset.IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 2 
    )

    valid_dataset = dataset.IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 1 
    )

    vector_generating_model = load_vectors('/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/models/imdb_embedding.bin')
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, vector_generating_model) # dict --> key(index),value(word vector)

    device = torch.device("cuda")

    model = lstm.LSTM(embedding_matrix)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_accuracy = 0
    early_stopping_counter = 0

    for epoch in range(config.EPOCHS):
        engine.train(train_data_loader, model, optimizer, device)
        outputs, targets = engine.evaluate(valid_data_loader, model, device)

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Fold : {fold}, Epoch:{epoch}, Accuracy Score :{accuracy}")

        # Early Stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break

if __name__ == '__main__':
    df = pd.read_csv("/home/praveen/Desktop/Projects/Approching_Almost_Any_ML_Prob_Book/NLP/data/IMDB_folds.csv")
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)


