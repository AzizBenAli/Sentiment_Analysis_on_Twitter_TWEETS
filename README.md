## Sentiment Analysis with Bidirectional LSTM, BERT, and GPT-2
This project implements sentiment analysis using three different models: Bidirectional LSTM, BERT (Bidirectional Encoder Representations from Transformers), and GPT-2 (Generative Pre-trained Transformer 2). The goal is to compare the performance of these models and determine which one performs better for sentiment analysis on the given dataset.

## Dataset
The dataset used in this project consists of text data labeled with sentiment classes (e.g., positive, negative, neutral). The dataset is split into a training set and a test set. The training set is used to train the models, and the test set is used to evaluate their performance.

## Models
- Bidirectional LSTM
The Bidirectional LSTM model is a deep learning model that uses Long Short-Term Memory (LSTM) cells in both forward and backward directions. It is well-suited for sequence data like text due to its ability to capture long-term dependencies.

- BERT (Bidirectional Encoder Representations from Transformers)
BERT is a state-of-the-art pre-trained language model developed by Google. It uses a transformer architecture and bidirectional training to learn contextualized word embeddings. We fine-tune the pre-trained BERT model for sentiment analysis on our dataset.

- Naive Bayesian Classification

## Implementation
The project is implemented in Python using popular deep learning libraries, including PyTorch and Hugging Face Transformers. The steps of the implementation are as follows:

Data Preprocessing: The text data is tokenized and converted into numerical sequences suitable for feeding into the models.

Bidirectional LSTM Model: The bidirectional LSTM model is defined and trained using the training set.

BERT Model: The pre-trained BERT model is loaded, and a classification head is added to fine-tune the model on the training set.

Evaluation: The performance of each model is evaluated using the test set.

## Results
The results of the evaluation are presented and compared for each model. We analyze the accuracy and other metrics to determine which model performs better for sentiment analysis on our dataset.

## Conclusion
In conclusion, this project explores the performance of three different models (Bidirectional LSTM, BERT, and NB) for sentiment analysis. The results will help us understand which model is most suitable for our specific task and dataset.

## Usage
To run the code and reproduce the results, follow the instructions in the main.ipynb notebook. Make sure you have all the required libraries and dependencies installed.

## Acknowledgments
We acknowledge the authors of the Bidirectional LSTM model, the BERT model, and the NB model for their groundbreaking work and the developers of the PyTorch and Hugging Face Transformers libraries for providing powerful tools for natural language processing.
