import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def load_pair(path):
    vocab = dict()
    for word1, word2 in pd.read_csv(path).values:
        if word1 not in vocab:
            vocab[word1] = word2
        if word2 not in vocab:
            vocab[word2] = word1
    return vocab


def load_poly(path):
    vocab = dict()
    for word, cand_str in pd.read_csv(path).values:
        if word not in vocab:
            vocab[word] = set()
        cands = cand_str.split('/')
        vocab[word].update(cands)
    return vocab


def sent2label(pairs):
    label = list()
    for pair in pairs:
        label.append(pair['label'])
    return label


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
