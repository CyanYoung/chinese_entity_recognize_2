import json
import pickle as pk

from sklearn_crfsuite.metrics import flat_f1_score, flat_accuracy_score

from recognize import predict


seq_len = 100

path_label_ind = 'feat/label_ind.pkl'
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

slots = list(label_inds.keys())
slots.remove('N')
slots.remove('O')


def align(sents):
    align_texts = list()
    label_mat = list()
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


def test(name, phase, align_texts, label_mat):
    pred_mat = list()
    for text in align_texts:
        pairs = predict(text, name, phase)
        preds = [pred for word, pred in pairs]
        pred_mat.append(preds)
    f1 = flat_f1_score(label_mat, pred_mat, average='weighted', labels=slots)
    print('\n%s %s %.2f' % (name, ' f1:', f1))
    print('%s %s %.2f' % (name, 'acc:', flat_accuracy_score(label_mat, pred_mat)))


if __name__ == '__main__':
    prefix = 'data/general/'
    path = prefix + 'test.json'
    with open(path, 'r') as f:
        sents = json.load(f)
    align_texts, label_mat = align(sents)
    test('rnn', 'special', align_texts, label_mat)
    test('rnn_crf', 'special', align_texts, label_mat)
