import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
# from keras.utils.vis_utils import plot_model
from imblearn.over_sampling import SMOTE

import nltk
import numpy as np

# The maximum number of words to be used. (most frequent)
from tensorflow.keras.models import load_model

MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 32
# Stop words
stopwords_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                  "itself", "they", "them", "their", "theirs", "themselves", "which", "who", "whom", "these",
                  "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                  "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "against", "into", "through", "during",
                  "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                  "under", "again", "further", "then", "once", "here", "there", "when", "why", "how", "all", "any",
                  "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                  "same", "so", "than", "too", "very", "s", "t", "don", "should", "now"]


class SentimentAnalyzer:

    def __init__(self):
        self.df, self.tokenizer = self.pre_initialize()

    def import_and_prepare(self, filepath):
        df = pd.read_csv(filepath, names=['sentence', 'operation'], sep=',', engine='python')
        # df = shuffle(df)
        sentences = df['sentence'].values
        y = df['operation'].values
        return df, sentences, y

    def filter_stopwords(self, sentences, stopwords_list):
        stopwords_set = set(stopwords_list)
        filtered = []
        for sentence in sentences:
            tokenized_sentence = word_tokenize(sentence)
            filtered_sentence = []
            for w in tokenized_sentence:
                if w not in stopwords_set:
                    filtered_sentence.append(w)
            filtered.append(filtered_sentence)
        return filtered

    def detokenize(self, filtered_sentences):
        detokenized_sentences = []
        for sentence in filtered_sentences:
            detokenized_sentences.append(TreebankWordDetokenizer().detokenize(sentence))
        return detokenized_sentences

    def plot_history(self, history):
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def plot_label_distribution(self, dataframe):
        dataframe['operation'].value_counts().plot(kind="bar")

    def init_tokenizer(self, MAX_NB_WORDS, dataframe):
        tokenizer = Tokenizer(MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(dataframe['filtered_sentence'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return tokenizer

    def create_model(self, max_words, embedding_dimensions, X):
        model = Sequential()
        model.add(Embedding(max_words, embedding_dimensions, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def lstm_train(self, df, tokenizer, max_sequence_length, embedding_dimensions):
        X = tokenizer.texts_to_sequences(df['filtered_sentence'].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X.shape)
        Y = pd.get_dummies(df['operation']).values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        # Oversampling the minority class
        smote = SMOTE('minority')
        X_train, Y_train = smote.fit_sample(X_train, Y_train)

        model = self.create_model(max_sequence_length, embedding_dimensions, X)
        epochs = 150
        batch_size = 100
        history = model.fit(X_train, Y_train,
                            epochs=epochs, batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        accr = model.evaluate(X_test, Y_test)
        print(model.summary())
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        # plot_model(model, to_file='model.png')
        return model, history

    def infer(self, sentence, tokenizer, model):
        sentence_as_array = [sentence]
        filtered_commands = self.filter_stopwords(sentence_as_array, stopwords_list)
        seq = tokenizer.texts_to_sequences(filtered_commands)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        return pred

    def pre_initialize(self):
        df, sentences, y = self.import_and_prepare('tc_data/dataset_new.txt')
        # df_temp, sentences_temp, y_temp = import_and_prepare('data/dataset_new.txt')
        self.plot_label_distribution(df)
        filtered_sentences = self.filter_stopwords(sentences, stopwords_list)
        detokenized_sentences = self.detokenize(filtered_sentences)
        df['filtered_sentence'] = detokenized_sentences
        tokenizer = self.init_tokenizer(MAX_NB_WORDS, df)
        return df, tokenizer

    def get_sentiment(self, command, model):
        new_command = [command]
        filtered_commands = self.filter_stopwords(new_command, stopwords_list)
        seq = self.tokenizer.texts_to_sequences(filtered_commands)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)

        labels = ['Locate', 'Describe', 'No_Op']
        print("Predicted vector: ", pred, " Predicted Class: ", labels[np.argmax(pred)])


if __name__ == '__main__':
    # df, sentences, y = import_and_prepare('data/dataset.txt')
    # nltk.download('punkt')

    # df, tokenizer = pre_initialize()
    # model, history = lstm_train(df, tokenizer, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    # model.save('lstm.h5')
    # plot_history(history)

    sentimentAnalyzer = SentimentAnalyzer()

    # ====== Test ========
    model = load_model('./lstm.h5')
    new_command = 'What is this pen'
    sentimentAnalyzer.get_sentiment(new_command, model)
