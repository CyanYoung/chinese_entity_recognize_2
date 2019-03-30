import json
import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from util import sent2label


embed_len = 200
max_vocab = 5000
win_len = 7
seq_len = 50

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'


def embed(sents, path_word2ind, path_word_vec, path_embed):
    texts = sents.keys()
    model = Tokenizer(num_words=max_vocab, filters='', char_level=True)
    model.fit_on_texts(texts)
    word_inds = model.word_index
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def label2ind(sents, path_label_ind):
    labels = list()
    for pairs in sents.values():
        labels.extend(sent2label(pairs))
    labels = sorted(list(set(labels)))
    label_inds = dict()
    label_inds['N'] = 0
    for i in range(len(labels)):
        label_inds[labels[i]] = i + 1
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def add_buf(seqs):
    buf = [0] * int((win_len - 1) / 2)
    buf_seqs = list()
    for seq in seqs.tolist():
        buf_seqs.append(buf + seq + buf)
    return np.array(buf_seqs)


def align_sent(sents, path_sent, extra):
    texts = sents.keys()
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(texts)
    pad_seqs = pad_sequences(seqs, maxlen=seq_len)
    if extra:
        pad_seqs = add_buf(pad_seqs)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)


def align_label(sents, path_label):
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    ind_mat = list()
    for pairs in sents.values():
        inds = list()
        for pair in pairs:
            inds.append(label_inds[pair['label']])
        ind_mat.append(inds)
    pad_inds = pad_sequences(ind_mat, maxlen=seq_len)
    with open(path_label, 'wb') as f:
        pk.dump(pad_inds, f)


def vectorize(path_data, path_cnn_sent, path_rnn_sent, path_label):
    with open(path_data, 'r') as f:
        sents = json.load(f)
    embed(sents, path_word2ind, path_word_vec, path_embed)
    label2ind(sents, path_label_ind)
    align_sent(sents, path_cnn_sent, extra=True)
    align_sent(sents, path_rnn_sent, extra=False)
    align_label(sents, path_label)


if __name__ == '__main__':
    path_data = 'data/train.json'
    path_cnn_sent = 'feat/cnn_sent_train.pkl'
    path_rnn_sent = 'feat/rnn_sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_cnn_sent, path_rnn_sent, path_label)
