#!/usr/bin/env python3

import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape, LSTM, TimeDistributed
from keras.optimizers import Adam
from keras import callbacks
from keras.utils import to_categorical

import json


class ResetState(callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        self.model.reset_states()


def generator(features, labels, batch_size, symbols):
    # Create empty arrays to contain batch of features and labels#
    idx = np.arange(len(features))
    np.random.shuffle(idx)
    offset = 0
    while True:
        if offset + batch_size > len(idx):
            np.random.shuffle(idx)
            offset = 0

        # choose random index in features
        index = idx[offset: offset+batch_size]
        batch_features = to_categorical(
            features[index], num_classes=symbols)
        batch_labels = to_categorical(
            labels[index], num_classes=symbols)
        offset += batch_size
        yield batch_features, batch_labels


def main():

    validation_split = 0.2

    with open('reddit_jokes.json', 'r') as f:
        data = json.loads(f.read())

    jokes = []
    for j in data:
        content = j['title'] + ' ' + j['body']
        content = ''.join(map(lambda c: c if ord(c) < 128 else ' ', content))
        content = content.replace('\n', ' ')
        content = content.strip()
        content = ' '.join(content.split(' '))
        if len(content) > 200:
            continue
        jokes.append(content)
    print('jokes: ', len(jokes))

    everything = sorted(set(''.join(jokes)))
    everything = filter(lambda c: ord(c) < 128, everything)
    char_map = {}
    reverse_char_map = {}
    for i, c in enumerate(everything):
        char_map[c] = i+2
        reverse_char_map[i+2] = c
    symbols = len(char_map)+2

    print('symbols: ', symbols)

    features = list(
        map(lambda j: np.array([1] + list(map(lambda c: char_map.get(c, char_map[' ']), j)) + [0], dtype='int8'), jokes))

    largest = max(map(len, features))
    x = np.zeros([len(features), largest], dtype='int8')
    y = np.zeros([len(features), largest], dtype='int8')

    for i, f in enumerate(features):
        x[i, :len(f)] = f
        y[i, :len(f)-1] = f[1:]

    idx = np.arange(len(x))
    np.random.shuffle(idx)

    val_samples = int(validation_split * len(x))

    val_idx = idx[:val_samples]
    val_x = to_categorical(x[val_idx], num_classes=symbols)
    val_y = to_categorical(y[val_idx], num_classes=symbols)

    train_idx = idx[val_samples:]
    x = x[train_idx]
    y = y[train_idx]

    batch_size = 64

    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=[largest, symbols]))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(symbols, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['categorical_accuracy'])

    print(model.summary())

    model.fit_generator(
        generator(x, y, batch_size, symbols), steps_per_epoch=len(x) // batch_size,
        epochs=20, shuffle=False, validation_data=(val_x, val_y))

    print("generating jokes...")

    for j in range(1000):
        gen = np.zeros([1, largest, symbols], dtype='int8')
        gen[0][0][1] = 1
        out = ''
        for i in range(1, 200):
            pred = model.predict(gen)

            pred = pred[0][i-1]

            letter = np.random.choice(symbols, p=pred)

            if letter == 0 or letter == 1:
                break
            out += reverse_char_map[letter]
            gen[0][i][letter] = 1
        print(out)


if __name__ == '__main__':
    main()
