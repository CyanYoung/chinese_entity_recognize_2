import json
import pickle as pk

from sklearn.metrics import f1_score, accuracy_score

from recognize import predict


seq_len = 100

path_sent = 'data/general/test.json'
path_label_ind = 'feat/label_ind.pkl'
with open(path_sent, 'r') as f:
    sents = json.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

slots = list(label_inds.keys())
slots.remove('N')
slots.remove('O')


def align(sents):
    align_texts, label_mat = list(), list()
    for text, quaples in sents.items():
        labels = list()
        for quaple in quaples:
            labels.append(quaple['label'])
        while len(text) > seq_len:
            trunc_text = text[:seq_len]
            trunc_labels = labels[:seq_len]
            align_texts.append(trunc_text)
            label_mat.append(trunc_labels)
            text = text[seq_len:]
            labels = labels[seq_len:]
        align_texts.append(text)
        label_mat.append(labels)
    return align_texts, label_mat


def flat(labels):
    flat_labels = list()
    for label in labels:
        flat_labels.extend(label)
    return flat_labels


def test(name, phase, align_texts, label_mat):
    pred_mat = list()
    for text in align_texts:
        pairs = predict(text, name, phase)
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
    labels, preds = flat(label_mat), flat(pred_mat)
    f1 = f1_score(labels, preds, average='weighted', labels=slots)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(labels, preds)))


if __name__ == '__main__':
    align_texts, label_mat = align(sents)
    test('rnn', 'special', align_texts, label_mat)
    test('rnn_crf', 'special', align_texts, label_mat)
