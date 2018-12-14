import json

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def count(path_freq, items, field, name):
    pairs = Counter(items)
    sort_items = [item for item, freq in pairs.most_common()]
    sort_freqs = [freq for item, freq in pairs.most_common()]
    item_freq = dict()
    for item, freq in zip(sort_items, sort_freqs):
        item_freq[item] = freq
    with open(path_freq, 'w') as f:
        json.dump(item_freq, f, ensure_ascii=False, indent=4)
    plot_freq(sort_items, sort_freqs, field, name, u_bound=20)


def plot_freq(items, freqs, field, name, u_bound):
    inds = np.arange(len(items))
    plt.title(name)
    plt.bar(inds[:u_bound], freqs[:u_bound], width=0.5)
    plt.xlabel(field)
    plt.ylabel('freq')
    plt.xticks(inds[:u_bound], items[:u_bound], rotation='vertical')
    plt.show()


def statistic(paths, name):
    with open(paths['train'], 'r') as f:
        sents = json.load(f)
    texts = sents.keys()
    slots = list()
    for quaples in sents.values():
        for quaple in quaples:
            if quaple['label'] != 'O':
                slots.append(quaple['label'])
    text_str = ''.join(texts)
    text_lens = [len(text) for text in texts]
    count(paths['vocab_freq'], text_str, 'vocab', name)
    count(paths['len_freq'], text_lens, 'text_len', name)
    count(paths['slot_freq'], slots, 'slot', name)
    print('%s slot_per_sent: %d' % (name, int(len(slots) / len(texts))))


if __name__ == '__main__':
    paths = dict()
    prefix = 'stat/general/'
    paths['train'] = 'data/general/train.json'
    paths['vocab_freq'] = prefix + 'vocab_freq.json'
    paths['len_freq'] = prefix + 'len_freq.json'
    paths['slot_freq'] = prefix + 'slot_freq.json'
    statistic(paths, 'general')
    prefix = 'stat/special/'
    paths['train'] = 'data/special/train.json'
    paths['vocab_freq'] = prefix + 'vocab_freq.json'
    paths['len_freq'] = prefix + 'len_freq.json'
    paths['slot_freq'] = prefix + 'slot_freq.json'
    statistic(paths, 'special')
