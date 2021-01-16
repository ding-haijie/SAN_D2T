import re
import json
from collections import Counter


def counter():
    """ Count the number of field_labels and words """
    threshold_field = 100
    most_frequent_words = 20000

    FIELD, WORD = [], []
    with open('../data/original/train.box', 'r') as f:
        box_list = f.readlines()
    with open('../data/original/train.sent', 'r') as f:
        sent_list = f.readlines()

    for box in box_list:
        field_list = box.strip().split('\t')
        for field in field_list:
            f_label = field.split(':')[0]
            f_label = re.sub(r'_[1-9]\d*$', '', f_label)
            f_label = re.sub(r'^\'+|^_+|^\|+', '', f_label)
            if f_label.strip() == '':
                continue
            FIELD.append(f_label)
    FIELD = Counter(FIELD)
    # print(FIELD.most_common(10))
    FIELD = {x: FIELD[x] for x in FIELD if FIELD[x] >= threshold_field}
    print(len(FIELD))  # 1664

    for sent in sent_list:
        text = sent.strip().split()
        for t in text:
            t = re.sub(r'``', '', t)
            t = re.sub(r'\'\'', '', t)
            if t.strip() == '':
                continue
            WORD.append(t)
    WORD = Counter(WORD)
    # print(WORD.most_common(10))
    WORD = WORD.most_common(most_frequent_words)  # most frequent 20000 words

    with open('../data/field_cnt.txt', 'w') as f:
        for x in FIELD:
            f.write(x + '\t' + str(FIELD[x]) + '\n')
    with open('../data/word_cnt.txt', 'w') as f:
        for word_pair in WORD:
            f.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')


def build_vocab():
    """ build vocabulary for field_labels and words """
    with open('../data/field_cnt.txt', 'r', encoding='utf-8') as f:
        field_cnt = f.readlines()
    with open('../data/word_cnt.txt', 'r', encoding='utf-8') as f:
        word_cnt = f.readlines()

    word2ind = dict()
    word2ind['PAD_TOKEN'] = 0
    word2ind['UNK_TOKEN'] = 1
    word2ind['SOS_TOKEN'] = 2
    word2ind['EOS_TOKEN'] = 3
    idx = 4
    for word_pair in word_cnt:
        word = word_pair.split('\t')[0]
        if word not in word2ind:
            word2ind[word] = idx
            idx += 1

    field2ind = dict()
    field2ind['PAD_FIELD'] = 0
    field2ind['UNK_FIELD'] = 1
    idx = 2
    for field_pair in field_cnt:
        field = field_pair.split('\t')[0]
        if field not in field2ind:
            field2ind[field] = idx
            idx += 1

    assert len(word2ind) == len(word_cnt) + 4
    assert len(field2ind) == len(field_cnt) + 2
    vocab = {'word2ind': word2ind, 'field2ind': field2ind}

    with open('data/vocab.json', 'w') as f:
        json.dump(vocab, f)


def process_data():
    box_paths = ['../data/original/train.box',
                 '../data/original/valid.box', '../data/original/test.box']
    nb_paths = ['../data/original/train.nb',
                '../data/original/valid.nb', '../data/original/test.nb']
    sent_paths = ['../data/original/train.sent',
                  '../data/original/valid.sent', '../data/original/test.sent']
    data_path = ['../data/train_data.json',
                 '../data/valid_data.json', '../data/test_data.json']

    mixed_data = []
    for box_path, nb_path, sent_path in zip(box_paths, nb_paths, sent_paths):
        with open(box_path, 'r') as f:
            box_list = f.readlines()
        with open(nb_path, 'r') as f:
            nb_list = f.readlines()
        with open(sent_path, 'r') as f:
            sent_list = f.readlines()
        box_tmp, text_tmp, smaples_data = [], [], []
        for box in box_list:  # box contains per sample's info_box
            b_field_single, b_pos_single, b_value_single = [], [], []
            field_list = box.strip().split('\t')
            for f_item in field_list:
                if len(f_item.split(':')) > 2:
                    continue  # in case of field_item containing special content, e.g. url-links
                f_type, f_value = f_item.split(':')
                if '<none>' in f_value or f_value.strip() == '' or f_type.strip() == '':
                    continue
                f_label = re.sub(r'_[1-9]\d*$', '', f_type)
                f_label = re.sub(r'^\'+|^_+|^\|+', '', f_label)
                if f_label.strip() == "":
                    continue
                # f_label = corrector.correct(f_label)
                b_field_single.append(f_label)
                b_value_single.append(f_value)
                if re.search(r'_[1-9]\d*$', f_type):
                    f_pos = int(f_type.split('_')[-1])
                    b_pos_single.append(f_pos if f_pos <= 30 else 30)
                else:
                    b_pos_single.append(1)
            r_pos_single = reverse_pos(b_pos_single)
            box_tmp.append({'field': b_field_single,
                            'pos': b_pos_single,
                            'rpos': r_pos_single,
                            'value': b_value_single})
        line = 0
        for nb in nb_list:
            sent = sent_list[line].strip()
            sent = re.sub('``', '', sent)
            sent = re.sub('\'\'', '', sent)
            line += int(nb)
            text_tmp.append(sent)

        assert len(text_tmp) == len(box_tmp)

        for i in range(len(box_tmp)):
            smaples_data.append({'info_box': box_tmp[i], 'text': text_tmp[i]})

        mixed_data.append(smaples_data)

    for idx, data_p in enumerate(data_path):
        with open(data_p, 'w') as f:
            json.dump(mixed_data[idx], f)


def reverse_pos(box_pos):
    """ Reverse field_position """
    temp_pos, reversed_pos = [], []
    for pos in box_pos:
        if int(pos) == 1 and len(temp_pos) != 0:
            reversed_pos.extend(temp_pos[::-1])
            temp_pos = []
        temp_pos.append(pos)
    reversed_pos.extend(temp_pos[::-1])
    return reversed_pos


def check_process():
    file_paths = ['../data/train_data.json',
                  '../data/valid_data.json', '../data/test_data.json']
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for sample in data:
            info_box = sample['info_box']
            field, pos, rpos, value = info_box.items()
            assert len(field[1]) == len(pos[1]) == len(
                rpos[1]) == len(value[1])


if __name__ == '__main__':
    counter()
    build_vocab()
    process_data()
    check_process()
