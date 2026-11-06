# 🎬 Movie Reviews Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that uses Deep Learning and LSTM networks to classify movie reviews as **Positive** or **Negative**.

The project goes beyond model development by implementing a modular Python architecture, a structured text preprocessing pipeline, experiment tracking with MLflow, and model checkpointing.

---

## 🎯 Project Overview

Sentiment Analysis is a Natural Language Processing task that aims to identify the emotional tone expressed in textual data.

In this project, I developed an LSTM-based Deep Learning model to analyze movie reviews and predict their sentiment.

The main objective was not only to train an accurate model but also to apply better Machine Learning engineering practices by transforming a notebook-based workflow into a structured, reusable Python project.

### Key Features

- Text preprocessing and cleaning using NLTK.
- Custom Vocabulary implementation with special tokens.
- Text encoding and sequence padding.
- LSTM-based Deep Learning architecture using TensorFlow/Keras.
- Early Stopping to reduce overfitting.
- Model Checkpointing to preserve the best-performing model.
- Experiment tracking and metrics logging using MLflow.
- Modular and maintainable project structure.

---

## 🧠 Machine Learning Pipeline

The project follows these main stages:

### 1. Data Loading
- Load movie reviews from a CSV dataset.
- Separate review texts and sentiment labels.
- Split the dataset into training, validation, and testing sets.

### 2. Text Preprocessing

Using NLTK and Python:

- Convert text to lowercase.
- Remove HTML tags.
- Remove unnecessary punctuation and special characters.
- Tokenize text.
- Remove stopwords while preserving important negation words.
- Apply WordNet Lemmatization.

### 3. Vocabulary Building

Implemented a custom `Vocabulary` class to:

- Build a word-to-index mapping.
- Filter words based on minimum frequency.
- Limit vocabulary size.
- Handle unknown words using an `<UNK>` token.
- Reserve special tokens such as `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>`.

The vocabulary is built using training data only to avoid data leakage.

### 4. Text Encoding & Padding

- Convert tokens into numerical sequences.
- Replace unknown words with the `<UNK>` index.
- Apply sequence padding and truncation.
- Use a fixed maximum sequence length.

### 5. Model Development

The model is built using TensorFlow/Keras.

**Architecture:**

- Embedding Layer
- LSTM Layer
- Dense Layer with ReLU activation
- Dropout Layer
- Output Layer with Sigmoid activation

The Embedding layer uses masking to prevent padded tokens from affecting the LSTM computations.

### 6. Model Training

The training process includes:

- Adam Optimizer.
- Binary Crossentropy Loss.
- Accuracy metric.
- Early Stopping.
- Model Checkpointing.

Early Stopping monitors validation loss and restores the best model weights to help reduce overfitting.

### 7. Model Evaluation

The model is evaluated using an independent test set.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📊 Experiment Tracking with MLflow

MLflow is integrated into the training workflow to track and manage model experiments.

It provides visibility into model performance and makes it easier to compare different training configurations.

Tracked information includes:

- Model hyperparameters.
- Training and validation metrics.
- Model artifacts.
- Training runs.
- Model summaries.

The MLflow UI is used to inspect experiments and compare runs.

### Launch MLflow UI

```bash
mlflow ui
```

Then open:

http://127.0.0.1:5000

---

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| NLTK | Text preprocessing |
| TensorFlow / Keras | Deep Learning and LSTM |
| Scikit-learn | Data splitting and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| PyYAML | Configuration management |
| MLflow | Experiment tracking |
| uv | Python package and environment management |

---

## 📁 Project Structure

```text
Movie-Reviews-Sentiment-Analysis/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── mlruns/
│
├── src/
│   ├── lstm.py
│   └── utils.py
│
├── config.yaml
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

### Code Organization

- `src/lstm.py`: Main execution script responsible for running the complete training pipeline.
- `src/utils.py`: Reusable functions and classes for data preprocessing, vocabulary building, model creation, training, and evaluation.
- `config.yaml`: Centralized configuration for dataset paths and model hyperparameters.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.13+
- uv
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

```bash
cd Movie-Reviews-Sentiment-Analysis
```

### 2. Install Dependencies

Using uv:

```bash
uv pip install pandas numpy nltk tensorflow scikit-learn matplotlib seaborn pyyaml mlflow tensorboard
```

### 3. Prepare the Dataset

Place your movie reviews CSV file in the appropriate data directory.

Make sure the dataset path in `config.yaml` points to the correct location.

### 4. Run the Project

```bash
uv run --no-project python ./src/lstm.py
```

---

## 📈 Experimental Results

The initial training experiment produced the following validation results:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | 88.16% |
| Best Validation Loss | 0.29936 |
| Best Epoch | 1 |
| Configured Maximum Epochs | 15 |
| Stopped Training At | Epoch 4 |

Early Stopping terminated training after validation performance stopped improving.

**Note:** These are initial validation results, not final test-set results. Final evaluation metrics will be reported separately after evaluating the best saved model on the test set.

---

## 🚀 Future Improvements

- Experiment with Bidirectional LSTM architectures.
- Compare LSTM with GRU and CNN-based text classification models.
- Perform hyperparameter tuning.
- Improve text preprocessing and vocabulary strategies.
- Experiment with pre-trained word embeddings such as GloVe.
- Add more comprehensive experiment comparisons through MLflow.
- Explore model deployment using FastAPI.
- Add automated testing and data validation.

---

## 💡 Key Learning Outcomes

Through this project, I focused on developing skills beyond model training:

- Designing modular and reusable ML code.
- Building an end-to-end NLP pipeline.
- Understanding sequence-based Deep Learning architectures.
- Managing experiments with MLflow.
- Applying regularization and callbacks to reduce overfitting.
- Moving from notebook experimentation toward structured Machine Learning engineering.

---

**Built with Python, TensorFlow, and MLflow.**