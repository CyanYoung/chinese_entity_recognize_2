import json

from flask import Flask, request

from argparse import ArgumentParser

from recognize import predict

from util import load_word, load_pair, load_triple, get_logger


path_slot = 'dict/slot.txt'
path_zh_en = 'dict/zh_en.csv'
path_label_key_slot = 'dict/label_key_slot.csv'
slots = load_word(path_slot)
zh_en = load_pair(path_zh_en)
label_key_slot = load_triple(path_label_key_slot)

app = Flask(__name__)

parser = ArgumentParser()
parser.add_argument('-host', type=str, default='127.0.0.1')
parser.add_argument('-port', type=str, default=2001)
args = parser.parse_args()

path_log_dir = 'log'
logger = get_logger('recognize', path_log_dir)


def init_entity(slots):
    entitys = dict()
    for slot in slots:
        entitys[slot] = list()
    return entitys


def map_slot(word, pred):
    for label, key, slot in label_key_slot:
        if pred == label:
            if key in word:
                return slot
    return pred


def insert(entity, label, fill_slots, entitys):
    slot = map_slot(entity, zh_en[label])
    fill_slots.append(slot)
    entitys[slot].append(entity)


def merge(pairs):
    entitys = init_entity(slots)
    fill_slots = list()
    entity = ''
    label = ''
    for word, pred in pairs:
        if pred[0] == 'B':
            if entity:
                insert(entity, label, fill_slots, entitys)
            entity = word
            label = pred[2:]
        elif pred[0] == 'O':
            if entity:
                insert(entity, label, fill_slots, entitys)
                entity = ''
        else:
            entity = entity + word
    if entity:
        insert(entity, label, fill_slots, entitys)
    return entitys, fill_slots


@app.route('/recognize', methods=['POST'])
def response():
    data = request.get_json()
    pairs = predict(data['content'], 'rnn_crf', 'special')
    entitys, fill_slots = merge(pairs)
    data['intent'] = '_'.join(fill_slots)
    data['entity'] = entitys
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    app.run(args.host, int(args.port), debug=True)
