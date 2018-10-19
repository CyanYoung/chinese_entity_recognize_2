from keras.layers import Dense, LSTM, Dropout
from keras.layers import TimeDistributed, Bidirectional


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(class_num, activation='softmax')
    ta = TimeDistributed(da)
    x = ba(embed_input)
    x = Dropout(0.5)(x)
    return ta(x)


def rnn_crf(embed_input, crf):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    x = ba(embed_input)
    x = Dropout(0.5)(x)
    return crf(x)
