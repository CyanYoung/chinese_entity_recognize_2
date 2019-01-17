import pickle as pk

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from keras_contrib.layers import CRF

from nn_arch import rnn, rnn_crf

from util import map_item


def define_nn_crf(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    crf = CRF(class_num)
    output = func(embed_input, crf)
    return Model(input, output)


def load_nn_crf(name, embed_mat, seq_len, class_num, phase):
    model = define_nn_crf(name, embed_mat, seq_len, class_num)
    model.load_weights(map_item('_'.join([phase, name]), paths))
    return model


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


seq_len = 100

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

funcs = {'rnn': rnn,
         'rnn_crf': rnn_crf}

paths = {'general_rnn': 'model/general/rnn.h5',
         'general_rnn_crf': 'model/general/rnn_crf.h5',
         'special_rnn': 'model/special/rnn.h5',
         'special_rnn_crf': 'model/special/rnn_crf.h5'}

models = {'general_rnn': load_model(map_item('general_rnn', paths)),
          'general_rnn_crf': load_nn_crf('rnn_crf', embed_mat, seq_len, len(label_inds), 'general'),
          'special_rnn': load_model(map_item('special_rnn', paths)),
          'special_rnn_crf': load_nn_crf('rnn_crf', embed_mat, seq_len, len(label_inds), 'special')}


def predict(text, name, phase):
    text = text.strip()
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item('_'.join([phase, name]), models)
    probs = model.predict(pad_seq)[0]
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds[-len(text):]]
    pairs = list()
    for word, pred in zip(text, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('rnn: %s' % predict(text, 'rnn', 'special'))
        print('rnn_crf: %s' % predict(text, 'rnn_crf', 'special'))
