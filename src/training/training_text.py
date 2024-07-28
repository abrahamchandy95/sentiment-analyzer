import os
import numpy as np
import pandas as pd
from keras import (
    layers, models, losses, optimizers, metrics, callbacks, regularizers
)

from data_vectorization import GloveVectorizer
from utils import split_data
from config import datasets_dir, PROJECT_DIR

def load_data():
    text_dir = os.path.join(datasets_dir, 'SENTIMENT140')
    train_data_path = os.path.join(
        text_dir, 'training.1600000.processed.noemoticon.csv'
    )
    test_data_path = os.path.join(
        text_dir, 'testdata.manual.2009.06.14.csv'
    )

    column_names = ['sentiment', 'tweet_id', 'date', 'query', 'user', 'tweet']
    train_raw = pd.read_csv(
        train_data_path, names=column_names, header=None, encoding='ISO-8859-1'
    )
    test_raw = pd.read_csv(
        test_data_path, names=column_names, header=None, encoding='ISO-8859-1'
    )

    train_raw.set_index('tweet_id', inplace=True)
    test_raw.set_index('tweet_id', inplace=True)

    train_df = train_raw.loc[:, ['tweet', 'sentiment']]
    test_df = test_raw.loc[:, ['tweet', 'sentiment']]
    return train_df, test_df

def df_to_X_y(dataframe, vectorizer):
    
    y = dataframe['sentiment'].to_numpy().astype(int)
    all_word_vector_sequences = []
    
    for message in dataframe['tweet']:
        message_as_vector_list = vectorizer.vectorize_text(message)
        all_word_vector_sequences.append(message_as_vector_list)
        
    all_word_vector_sequences = np.array(all_word_vector_sequences)
    return all_word_vector_sequences, y

def vectorize_data(train_df, vectorizer):
    
    train_data, _, val_data = split_data(train_df, test_size=0.1, val_size=0)

    X_train, y_train = df_to_X_y(train_data, vectorizer)
    X_val, y_val = df_to_X_y(val_data, vectorizer)

    normalizer = layers.Normalization()
    normalizer.adapt(X_train)

    # convert values to 0 and 1
    y_train = (y_train == 4).astype(int)
    y_val = (y_val == 4).astype(int)
    return X_train, y_train, X_val, y_val, normalizer

def build_and_train_model(
    X_train, y_train, X_val, y_val, vector_dims, normalizer
):
    max_text_length = 120
    model = models.Sequential([
        layers.Input(shape=(max_text_length, vector_dims)),
        layers.Masking(mask_value=0.0),
        normalizer,
        layers.LSTM(
            64, 
            return_sequences=True, 
            kernel_regularizer=regularizers.l2(0.01)
        ),
        layers.Dropout(0.3),
        layers.LSTM(
            64, 
            return_sequences=True, 
            kernel_regularizer=regularizers.l2(0.01)
        ),
        layers.Dropout(0.3),
        layers.LSTM(
            64, 
            kernel_regularizer=regularizers.l2(0.01)
        ),
        layers.Dropout(0.3),
        layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_regularizer=regularizers.l2(0.01)
        )
    ])
    
    chekcpoint_dir = os.path.join(PROJECT_DIR, 'models')
    os.makedirs(chekcpoint_dir, exist_ok=True) 

    cp = callbacks.ModelCheckpoint(
        os.path.join(chekcpoint_dir, 'best_model_text.keras'), 
        save_best_only=True
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    
    model.fit(
        X_train, 
        y_train, 
        validation_data=(X_val, y_val), 
        epochs=30, 
        callbacks=[cp, early_stopping]
    )
    
    return model

    
def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test).flatten()
    # test data has 3 choices
    y_test_converted = np.where(
        y_test == 4, 1, np.where(y_test == 2, -1, 0)
    )
    y_preds = np.where(
        predictions > 0.51, 1, np.where(predictions < 0.49, 0, -1)
    )
    # calculate accuracy
    correct_preds = (y_preds == y_test_converted)
    accuracy = np.sum(correct_preds) / len(y_test_converted)
    print(f'Test Accuracy: {accuracy:.2%}')
    
if __name__ =='__main__':
    train_df, test_df = load_data()
    vectorizer = GloveVectorizer(datasets_dir)
    # training and val data
    X_train, y_train, X_val, y_val, normalizer = vectorize_data(
        train_df, vectorizer
    )
    
    vector_dims = vectorizer.vector_dims
    model = build_and_train_model(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        vector_dims=vector_dims, 
        normalizer=normalizer
    )
    
    # Prepare test data
    X_test, y_test = df_to_X_y(test_df, vectorizer=vectorizer)
    X_test = np.array(X_test, dtype=np.float32)
    evaluate_model(model, X_test, y_test)