import yaml
import mlflow

from utils import load_dataset, clean_split_data, Vocabulary ,build_model , train_model , classification_report_and_confusion_matrix , Text_pad_sequences



if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    path = config["data"]["dataset_path"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    max_vocab_size = config["data"]["max_vocab_size"]
    min_freq = config["data"]["min_freq"]
    max_len = config["data"]["max_len"]
    
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    
    embedding_dim = config["model"]["embedding_dim"]
    lstm_units = config["model"]["lstm_units"]
    dropout_rate = config["model"]["dropout_rate"]
    learning_rate = config["model"]["learning_rate"]
    model_save_path = config["model"]["model_save_path"]
    
    
    data = load_dataset(path)
    
    X_train_tokens, X_val_tokens, X_test_tokens, y_train, y_val, y_test = clean_split_data(data, test_size, val_size, random_seed=42)
    
    vocab = Vocabulary( min_freq , max_vocab_size )
    vocab.build(X_train_tokens)
    
    X_train_encoded = X_train_tokens.apply(vocab.encode)
    X_val_encoded = X_val_tokens.apply(vocab.encode)
    X_test_encoded = X_test_tokens.apply(vocab.encode)

    X_train_padded = Text_pad_sequences(X_train_encoded, maxlen=max_len)
    X_val_padded = Text_pad_sequences(X_val_encoded, maxlen=max_len)
    X_test_padded = Text_pad_sequences(X_test_encoded, maxlen=max_len)

    model = build_model(max_vocab_size, embedding_dim, lstm_units, dropout_rate , learning_rate)
    history , y_pred , run_id = train_model(model, X_train_padded, y_train, X_val_padded, y_val,X_test_padded, batch_size, epochs, early_stopping_patience, model_save_path)
    report, cm = classification_report_and_confusion_matrix(y_test ,y_pred, run_id)