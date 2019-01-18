import os

import json
import pandas as pd

import re

from random import shuffle, choice

from util import load_word, load_pair, load_poly


path_zh_en = 'dict/zh_en.csv'
path_pre_name = 'dict/pre_name.txt'
path_end_name = 'dict/end_name.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
zh_en = load_pair(path_zh_en)
pre_names = load_word(path_pre_name)
end_names = load_word(path_end_name)
homo_dict = load_poly(path_homo)
syno_dict = load_poly(path_syno)


def save(path, sents):
    with open(path, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)


def general_prepare(path_txt, path_json):
    sents = dict()
    pairs = list()
    with open(path_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pair = dict()
                word, label = line.split()
                pair['word'] = word
                pair['label'] = label
                pairs.append(pair)
            elif pairs:
                text = ''.join([pair['word'] for pair in pairs])
                sents[text] = pairs
                pairs = []
    save(path_json, sents)


def make_name(pre_names, end_names, num):
    names = list()
    for i in range(num):
        pre_name = choice(pre_names)
        end_name = choice(end_names)
        names.append(pre_name + end_name)
    return names


def dict2list(sents):
    word_mat, label_mat = list(), list()
    for pairs in sents.values():
        words, labels = list(), list()
        for pair in pairs:
            words.append(pair['word'])
            labels.append(pair['label'])
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def list2dict(word_mat, label_mat):
    sents = dict()
    for words, labels in zip(word_mat, label_mat):
        text = ''.join(words)
        pairs = list()
        for word, label in zip(words, labels):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def select(part):
    if part[0] == '[' and part[-1] == ']':
        word = part[1:-1]
        cands = set()
        cands.add(word)
        if word in syno_dict:
            cands.update(syno_dict[word])
        if word in homo_dict:
            cands.update(homo_dict[word])
        return choice(list(cands))
    elif part[0] == '(' and part[-1] == ')':
        word = part[1:-1]
        return choice([word, ''])
    else:
        return part


def generate(temps, slots, num):
    word_mat, label_mat = list(), list()
    for i in range(num):
        parts = choice(temps)
        words, labels = list(), list()
        for part in parts:
            if part in slots:
                entity = choice(slots[part])
                words.extend(entity)
                labels.append('B-' + part)
                if len(entity) > 1:
                    labels.extend(['I-' + part] * (len(entity) - 1))
            else:
                word = select(part)
                words.extend(word)
                labels.extend(['O'] * len(word))
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def sync_shuffle(list1, list2):
    pairs = list(zip(list1, list2))
    shuffle(pairs)
    return zip(*pairs)


def label_sent(path):
    sents = dict()
    for text, entity_str, label_str in pd.read_csv(path).values:
        entitys, labels = entity_str.split(), label_str.split()
        if len(entitys) != len(labels):
            print('skip: %s', text)
            continue
        slots = ['O'] * len(text)
        for entity, label in zip(entitys, labels):
            heads = [iter.start() for iter in re.finditer(entity, text)]
            entity_len = len(entity)
            for head in heads:
                tail = head + entity_len
                if slots[head:tail] != ['O'] * entity_len:
                    print('skip: %s in %s' % (entity, text))
                    continue
                slots[head] = 'B-' + label
                for i in range(1, entity_len):
                    slots[head + i] = 'I-' + label
        pairs = list()
        for word, label in zip(text, slots):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def expand(sents, gen_word_mat, gen_label_mat):
    word_mat, label_mat = dict2list(sents)
    word_mat.extend(gen_word_mat)
    label_mat.extend(gen_label_mat)
    word_mat, label_mat = sync_shuffle(word_mat, label_mat)
    bound1 = int(len(word_mat) * 0.7)
    bound2 = int(len(word_mat) * 0.9)
    train_sents = list2dict(word_mat[:bound1], label_mat[:bound1])
    dev_sents = list2dict(word_mat[bound1:bound2], label_mat[bound1:bound2])
    test_sents = list2dict(word_mat[bound2:], label_mat[bound2:])
    return train_sents, dev_sents, test_sents


def special_prepare(paths):
    temps = list()
    with open(paths['temp'], 'r') as f:
        for line in f:
            parts = line.strip().split()
            temps.append(parts)
    slots = dict()
    files = os.listdir(paths['slot_dir'])
    for file in files:
        label = zh_en[os.path.splitext(file)[0]]
        slots[label] = list()
        with open(os.path.join(paths['slot_dir'], file), 'r') as f:
            for line in f:
                slots[label].append(line.strip())
    names = make_name(pre_names, end_names, num=1000)
    slots['PER'].extend(names)
    gen_word_mat, gen_label_mat = generate(temps, slots, num=5000)
    with open(paths['fuse'], 'r') as f:
        sent1s = json.load(f)
    sent2s = label_sent(paths['extra'])
    sents = dict(sent1s, **sent2s)
    train_sents, dev_sents, test_sents = expand(sents, gen_word_mat, gen_label_mat)
    save(paths['train'], train_sents)
    save(paths['dev'], dev_sents)
    save(paths['test'], test_sents)


if __name__ == '__main__':
    prefix = 'data/general/'
    path_txt = prefix + 'train.txt'
    path_json = prefix + 'train.json'
    general_prepare(path_txt, path_json)
    path_txt = prefix + 'dev.txt'
    path_json = prefix + 'dev.json'
    general_prepare(path_txt, path_json)
    path_txt = prefix + 'test.txt'
    path_json = prefix + 'test.json'
    general_prepare(path_txt, path_json)
    paths = dict()
    paths['fuse'] = prefix + 'test.json'
    prefix = 'data/special/'
    paths['train'] = prefix + 'train.json'
    paths['dev'] = prefix + 'dev.json'
    paths['test'] = prefix + 'test.json'
    paths['temp'] = prefix + 'template.txt'
    paths['slot_dir'] = prefix + 'slot'
    paths['extra'] = prefix + 'extra.csv'
    special_prepare(paths)
