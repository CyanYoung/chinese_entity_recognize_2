import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from keras_contrib.layers import CRF

from nn_arch import rnn, rnn_crf

from util import map_item


batch_size = 32

path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

funcs = {'rnn': rnn,
         'rnn_crf': rnn_crf}

paths = {'general_rnn': 'model/general/rnn.h5',
         'general_rnn_crf': 'model/general/rnn_crf.h5',
         'special_rnn': 'model/special/rnn.h5',
         'special_rnn_crf': 'model/special/rnn_crf.h5',
         'rnn_plot': 'model/plot/rnn.png',
         'rnn_crf_plot': 'model/plot/rnn_crf.png'}


def load_feat(path_feats):
    with open(path_feats['sent_train'], 'rb') as f:
        train_sents = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['sent_dev'], 'rb') as f:
        dev_sents = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_sents, train_labels, dev_sents, dev_labels


def compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    if name == 'rnn_crf':
        crf = CRF(class_num)
        output = func(embed_input, crf)
        loss = crf.loss_function
        acc = crf.accuracy
    else:
        output = func(embed_input, class_num)
        loss = 'categorical_crossentropy'
        acc = 'accuracy'
    model = Model(input, output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=[acc])
    return model


def fit(name, epoch, embed_mat, label_inds, path_feats, phase):
    train_sents, train_labels, dev_sents, dev_labels = load_feat(path_feats)
    seq_len = len(train_sents[0])
    class_num = len(label_inds)
    model = compile(name, embed_mat, seq_len, class_num)
    if phase == 'special':
        model.load_weights(map_item('_'.join(['general', name]), paths))
    path = map_item('_'.join([phase, name]), paths)
    check_point = ModelCheckpoint(path, monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(train_sents, train_labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_data=(dev_sents, dev_labels))


if __name__ == '__main__':
    path_feats = dict()
    prefix = 'feat/general/'
    path_feats['sent_train'] = prefix + 'sent_train.pkl'
    path_feats['label_train'] = prefix + 'label_train.pkl'
    path_feats['sent_dev'] = prefix + 'sent_dev.pkl'
    path_feats['label_dev'] = prefix + 'label_dev.pkl'
    fit('rnn', 10, embed_mat, label_inds, path_feats, 'general')
    fit('rnn_crf', 10, embed_mat, label_inds, path_feats, 'general')
    prefix = 'feat/special/'
    path_feats['sent_train'] = prefix + 'sent_train.pkl'
    path_feats['label_train'] = prefix + 'label_train.pkl'
    path_feats['sent_dev'] = prefix + 'sent_dev.pkl'
    path_feats['label_dev'] = prefix + 'label_dev.pkl'
    fit('rnn', 10, embed_mat, label_inds, path_feats, 'special')
    fit('rnn_crf', 10, embed_mat, label_inds, path_feats, 'special')
