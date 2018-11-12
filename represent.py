import json
import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from util import sent2label


embed_len = 200
max_vocab = 5000
seq_len = 100

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


def align_sent(sents, path_sent):
    texts = sents.keys()
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(texts)
    align_seqs = list()
    for seq in seqs:
        while len(seq) > seq_len:
            trunc_seq = seq[:seq_len]
            align_seqs.append(trunc_seq)
            seq = seq[seq_len:]
        pad_seq = pad_sequences([seq], maxlen=seq_len)[0]
        align_seqs.append(pad_seq)
    align_seqs = np.array(align_seqs)
    with open(path_sent, 'wb') as f:
        pk.dump(align_seqs, f)


def align_label(sents, path_label):
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    class_num = len(label_inds)
    ind_mat = list()
    for pairs in sents.values():
        inds = list()
        for pair in pairs:
            inds.append(label_inds[pair['label']])
        while len(inds) > seq_len:
            trunc_inds = inds[:seq_len]
            trunc_inds = to_categorical(trunc_inds, num_classes=class_num)
            ind_mat.append(trunc_inds)
            inds = inds[seq_len:]
        pad_inds = pad_sequences([inds], maxlen=seq_len)[0]
        pad_inds = to_categorical(pad_inds, num_classes=class_num)
        ind_mat.append(pad_inds)
    ind_mat = np.array(ind_mat)
    with open(path_label, 'wb') as f:
        pk.dump(ind_mat, f)


def merge_vectorize(path_general_train, path_special_train):
    with open(path_general_train, 'r') as f:
        general_sents = json.load(f)
    with open(path_special_train, 'r') as f:
        special_sents = json.load(f)
    sents = dict(general_sents, **special_sents)
    embed(sents, path_word2ind, path_word_vec, path_embed)
    label2ind(sents, path_label_ind)


def vectorize(paths):
    with open(paths['data'], 'r') as f:
        sents = json.load(f)
    align_sent(sents, paths['sent'])
    align_label(sents, paths['label'])


if __name__ == '__main__':
    path_general_train = 'data/general/train.json'
    path_special_train = 'data/special/train.json'
    merge_vectorize(path_general_train, path_special_train)
    paths = dict()
    prefix = 'feat/general/'
    paths['data'] = path_general_train
    paths['sent'] = prefix + 'sent_train.pkl'
    paths['label'] = prefix + 'label_train.pkl'
    vectorize(paths)
    paths['data'] = 'data/general/dev.json'
    paths['sent'] = prefix + 'sent_dev.pkl'
    paths['label'] = prefix + 'label_dev.pkl'
    vectorize(paths)
    prefix = 'feat/special/'
    paths['data'] = path_special_train
    paths['sent'] = prefix + 'sent_train.pkl'
    paths['label'] = prefix + 'label_train.pkl'
    vectorize(paths)
    paths['data'] = 'data/special/dev.json'
    paths['sent'] = prefix + 'sent_dev.pkl'
    paths['label'] = prefix + 'label_dev.pkl'
    vectorize(paths)
