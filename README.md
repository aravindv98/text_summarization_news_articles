# Text Summarization of News Articles

This repository provides an end-to-end pipeline for summarizing news articles using advanced Natural Language Processing (NLP) techniques. Two prominent deep learning architectures are explored: Long Short-Term Memory (LSTM) based encoder-decoder models and Transformer-based models. The project utilizes the [CNN/Daily Mail dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/code) from Kaggle for training and evaluating the summarization models.

## Overview

Text summarization involves condensing a long piece of text into a shorter version while preserving its key information and overall meaning. This project focuses on **abstractive summarization**, where the model generates novel sentences that may not appear in the original text instead of extractive methods that select and compile existing sentences.

## Features

- **Data Preprocessing**:
  - Loading and cleaning the CNN/Daily Mail dataset (e.g., removing duplicates, handling missing values).
  - Tokenization, lowercasing, and normalization of text data.
  - Splitting the dataset into training, validation, and test sets.
  - Implementing padding and truncation to ensure uniform input lengths for models.

- **LSTM-based Encoder-Decoder Model**:
  - Sequence-to-sequence architecture with attention mechanism.
  - Implemented using frameworks like TensorFlow/Keras or PyTorch.
  - Capable of handling long input sequences to generate coherent summaries.

- **Transformer-based Model**:
  - Utilizes multi-head self-attention, positional encodings, and feed-forward neural networks.
  - Scales efficiently with longer documents and captures long-range dependencies.

- **Evaluation Metrics**:
  - **Accuracy**: Measures the proportion of correct predictions made by the model.
  - **Loss**: Quantifies the difference between the predicted summaries and the reference summaries during training.
  - **BLEU Score**: Evaluates the quality of the generated summaries by comparing them to reference summaries based on n-gram precision.

### Prerequisites

- **Programming Language**: Python 3.7+
- **Deep Learning Frameworks**:
  - TensorFlow/Keras or PyTorch
- **Data Handling**:
  - Pandas, NumPy
- **Utilities**:
  - scikit-learn (for metrics)
  - tqdm (for progress bars)
- **Evaluation**:
  - [BLEU Score Package](https://pypi.org/project/bleu-score/)

### Installation

1. **Clone the Repository**:
   git clone https://github.com/aravindv98/text_summarization_news_articles.git
