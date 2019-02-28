from keras.layers import Dense, LSTM, Dropout, Bidirectional


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(class_num, activation='softmax')
    x = ba(embed_input)
    x = Dropout(0.2)(x)
    return da(x)


def rnn_crf(embed_input, crf):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    x = ba(embed_input)
    x = Dropout(0.2)(x)
    return crf(x)
