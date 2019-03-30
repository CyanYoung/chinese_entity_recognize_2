from keras.layers import Conv1D, LSTM, Dense, Bidirectional, Dropout, Multiply


win_len = 7


def cnn(embed_input, class_num):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid')
    da1 = Dense(200, activation='relu')
    da2 = Dense(class_num, activation='softmax')
    g = gate(embed_input)
    x = conv(embed_input)
    x = Multiply()([x, g])
    x = da1(x)
    x = Dropout(0.2)(x)
    return da2(x)


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(class_num, activation='softmax')
    x = ba(embed_input)
    x = Dropout(0.2)(x)
    return da(x)
