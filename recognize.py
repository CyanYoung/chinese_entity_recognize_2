import pickle as pk

import numpy as np

from keras.preprocessing.sequence import pad_sequences

from build import load_model

from util import map_item


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

ind_labels = dict()
for label, ind in label_inds.items():
    ind_labels[ind] = label

models = {'general_rnn': load_model('rnn', embed_mat, seq_len, len(label_inds), 'general'),
          'general_rnn_crf': load_model('rnn_crf', embed_mat, seq_len, len(label_inds), 'general'),
          'special_rnn': load_model('rnn', embed_mat, seq_len, len(label_inds), 'special'),
          'special_rnn_crf': load_model('rnn_crf', embed_mat, seq_len, len(label_inds), 'special')}


def predict(text, name, phase):
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
