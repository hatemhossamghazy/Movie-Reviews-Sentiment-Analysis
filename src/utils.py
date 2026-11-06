from sklearn.model_selection import train_test_split

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras import Sequential , regularizers
from tensorflow.keras.layers import (
    Input,
    SpatialDropout1D,
    Embedding,
    LSTM,
    Dropout,
    Dense
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score , accuracy_score
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yaml
import nltk
import mlflow
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

def load_dataset(path):
    data = pd.read_csv(path)
    return data

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))

    negation_words = {'no', 'not', 'nor', "don't", "doesn't", "didn't",
                    "isn't", "wasn't", "weren't", "won't", "wouldn't",
                    "couldn't", "shouldn't", "never"}

    stop_words -= negation_words

    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"[^a-zA-Z'\s]", ' ', text)
    tokens = text.split()
    # tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_split_data(data,test_size,val_size,random_seed=42):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    X = data['cleaned_text']
    Y = data['label']
    
    X_train , X_test , y_train , y_test = train_test_split(
    X ,
    Y ,
    test_size = test_size , 
    stratify = Y ,
    random_state=random_seed
    )

    X_train , X_val , y_train , y_val = train_test_split(
        X_train ,
        y_train ,
        test_size = val_size , 
        stratify = y_train , 
        random_state=random_seed
    )
    
    X_train_tokens = X_train.apply(lambda x: x.split())
    X_val_tokens = X_val.apply(lambda x: x.split())
    X_test_tokens = X_test.apply(lambda x: x.split())
    
    return X_train_tokens, X_val_tokens, X_test_tokens, y_train, y_val, y_test



class Vocabulary:
    def __init__(self, min_freq=2, max_vocab_size=20000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.word2idx = self.special_tokens.copy()
        self.idx2word = {
            idx: word
            for word, idx in self.word2idx.items()
        }

    def build(self, tokenized_texts):
        counter = Counter()

        for tokens in tokenized_texts:
            counter.update(tokens)

        # Filter and sort words
        words = [
            (word, freq)
            for word, freq in counter.items()
            if freq >= self.min_freq
        ]

        words.sort(key=lambda x: (-x[1], x[0]))

        # Respect maximum vocabulary size
        available_slots = (
            self.max_vocab_size - len(self.special_tokens)
        )

        for word, _ in words[:available_slots]:
            idx = len(self.word2idx)

            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens):
        return [
            self.word2idx.get(token, self.word2idx["<UNK>"])
            for token in tokens
        ]

    def decode(self, indices):
        return [
            self.idx2word.get(idx, "<UNK>")
            for idx in indices
        ]

    def __len__(self):
        return len(self.word2idx)
    
    
    
def Text_pad_sequences(sequences, maxlen , padding='post', truncating='post', value=0):
    return pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=padding,
        truncating=truncating,
        value=value
    )
    
    

def build_model(vocab_size, embedding_dim , lstm_units, dropout_rate  , learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim , mask_zero=True),
        tf.keras.layers.LSTM(lstm_units, dropout=dropout_rate, kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model



def train_model(model, X_train, y_train, X_val, y_val,X_test, batch_size, epochs, early_stopping_patience, model_save_path):
    mlflow.set_experiment("Movie_Reviews_Sentiment_Analysis_LSTM")
    with mlflow.start_run() as run:
        mlflow.tensorflow.autolog()
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint]
        )
        
        loss , accuracy = model.evaluate(X_val, y_val)
        y_pred = model.predict(X_test)
        
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_str = "\n".join(summary_lines)
        mlflow.log_text(summary_str, "model_summary.txt")
        
        mlflow.log_artifact("config.yaml")
        
        
        run_id = run.info.run_id
    return history , y_pred , run_id



def classification_report_and_confusion_matrix(
    y_true,
    y_pred,
    run_id
):
    y_pred_classes = (y_pred > 0.5).astype(int).ravel()

    report = classification_report(
        y_true,
        y_pred_classes,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred_classes,
        labels=[0, 1]
    )

    with mlflow.start_run(run_id=run_id):

        mlflow.log_text(
            report,
            "classification_report.txt"
        )

        mlflow.log_metric(
            "test_accuracy",
            accuracy_score(y_true, y_pred_classes)
        )

        mlflow.log_metric(
            "test_f1",
            f1_score(y_true, y_pred_classes)
        )

    return report, cm