import os
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras import models, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tweetsCleaner import tweet_cleaner, preprocessing_tweets

split = 3000
winner_model = 'nlpLSTMmodel'

# https://nlp.stanford.edu/projects/glove/
def embedding():
    words = {}
    with open('glove.twitter.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding_w = np.asarray(values[1:], dtype='float32')
            words[word] = embedding_w
    return words


def train(model: models.Sequential, x_train, y_train):
    y_train = to_categorical(np.asarray(y_train), 2)
    x_train = np.asarray(x_train)
    check_points = [ModelCheckpoint(winner_model, save_best_only=True, save_weights_only=False)]
    model.fit(x_train, y_train, batch_size=24, epochs=40, verbose=1, callbacks=check_points, validation_split=0.1)

    model.save(winner_model)


def predict(model: models.Sequential, x_test, y_test):
    # model.load_weights(winner_model)
    y_test = to_categorical(np.asarray(y_test), 2)
    x_test = np.asarray(x_test)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # PRINTS tests

    y_pred = model.predict(x_test)
    y_pred = [np.argmax(n) for n in y_pred]
    return y_pred


def make_model(first_layer):
    nlp_model = models.Sequential()
    nlp_model.add(first_layer)
    nlp_model.add(keras.layers.LSTM(128))
    nlp_model.add(keras.layers.LSTM(64))
    nlp_model.add(layers.Dropout(0.1))
    nlp_model.add(keras.layers.Dense(2, activation='sigmoid'))
    optimizer = keras.optimizers.Adam()

    nlp_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    return nlp_model



def token_tweets(tweets, labels, max_seq, embedding_dim):
    # Use Tokenizer of keras ->
    words = embedding()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)

    vocab_size = len(tokenizer.word_index) + 1
    encoded_tweets = tokenizer.texts_to_sequences(tweets)
    padded_tweets = pad_sequences(encoded_tweets, maxlen=max_seq, padding='post')
    embedding_mat = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        embedding_vector = words.get(word)
        if embedding_vector is not None:
            embedding_mat[i] = embedding_vector
        else:
            embedding_mat[i] = [0.0] * embedding_dim  # Unknown words!

    embedding_layer = layers.Embedding(vocab_size, embedding_dim,
                                       weights=[embedding_mat],
                                       input_length=max_seq,
                                       trainable=False)

    return embedding_layer, padded_tweets, vocab_size, embedding_layer


def cv_split(tweets, labels):
    x_train = tweets[:split]
    y_train = labels[:split]
    x_test = tweets[split:]
    y_test = labels[split:]
    return x_train, y_train, x_test, y_test


def run_model():
    tweets, labels = preprocessing_tweets()
    tweet_cleaner(tweets)
    embedding_layer, padded_tweets, vocab_size, labels = token_tweets(tweets, labels, max_seq=32, embedding_dim=256)
    x_train, y_train, x_test, y_test = cv_split(padded_tweets, labels)
    model = make_model(first_layer=embedding_layer)
    train(model, x_train, y_train)
    return predict(model, x_test, y_test)


if __name__ == '__main__':
    predicts = run_model()
    """
    Export the result and process them to predictions table.
    """
