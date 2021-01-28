import json
import numpy as np
import torch
import torch.utils.data as Data


class DataProcessor(object):
    def __init__(self, args):
        self.vocab = self._load_data('./data/vocab.json')
        self.word2ind = self.vocab['word2ind']
        self.ind2word = {key: value for value, key in self.word2ind.items()}
        self.word_vocab_size = len(self.word2ind)  # 20004

        self.field2ind = self.vocab['field2ind']
        self.ind2field = {key: value for value, key in self.field2ind.items()}
        self.field_vocab_size = len(self.field2ind)  # 1666

        self.PAD_TOKEN = self.word2ind['PAD_TOKEN']  # 0
        self.UNK_TOKEN = self.word2ind['UNK_TOKEN']  # 1
        self.SOS_TOKEN = self.word2ind['SOS_TOKEN']  # 2
        self.EOS_TOKEN = self.word2ind['EOS_TOKEN']  # 3
        self.PAD_FIELD = self.field2ind['PAD_FIELD']  # 0
        self.UNK_FIELD = self.field2ind['UNK_FIELD']  # 1

        # max number of the position
        self.pos_size = args.pos_size
        # max length of the summaries
        self.max_len = args.max_len
        # max number of the fields been chose
        self.max_field = args.max_field

        if args.train_mode:
            self.train_data = self._load_data('./data/train_data.json')
            self.dev_data = self._load_data('./data/valid_data.json')

            self.seq_info_train, self.seq_target_train = self._process_data(
                tag='train')
            self.seq_info_dev, self.seq_target_dev = self._process_data(
                tag='dev')
            self.test_data = self._load_data('./data/test_data.json')
        else:
            self.test_data = self._load_data('./data/test_data.json')

    def _process_data(self, tag):
        """
        process data to model-friendly format, i.e.
        input: (max_field, 4)  output: (max_len, 1)
        """
        if tag == 'train':
            tag_data = self.train_data
        elif tag == 'dev':
            tag_data = self.dev_data
        else:
            raise ValueError('illegal tag: ', tag)

        seq_info = np.zeros((len(tag_data), self.max_field, 4))  # PAD
        seq_target = np.zeros((len(tag_data), self.max_len))  # PAD
        for data_index, data_item in enumerate(tag_data):
            info_boxes = data_item['info_box']
            field, pos, rpos, value = info_boxes['field'], info_boxes[
                'pos'], info_boxes['rpos'], info_boxes['value']

            for idx, (f, p, rp, v) in enumerate(zip(field, pos, rpos, value)):
                if idx < self.max_field:
                    seq_info[data_index, idx,
                             0] = self.field2ind[f] if f in self.field2ind else self.UNK_FIELD
                    seq_info[data_index, idx, 1] = p
                    seq_info[data_index, idx, 2] = rp
                    seq_info[data_index, idx,
                             3] = self.word2ind[v] if v in self.word2ind else self.UNK_TOKEN
                else:
                    break
            tokens_text = data_item['text'].strip().split()
            seq_target[data_index, 0] = self.SOS_TOKEN
            for idx, token in enumerate(tokens_text):
                if (idx + 1) < self.max_len:
                    seq_target[data_index, idx + 1] = self.word2ind[
                        token] if token in self.word2ind else self.UNK_TOKEN
                else:
                    break
        return seq_info, seq_target

    def process_one_data(self, idx_data):
        """process data to model-friendly format one-by-one for test set."""
        seq_info = np.zeros((self.max_field, 4))  # PAD
        info_boxes = self.test_data[idx_data]['info_box']
        field, pos, rpos, value = info_boxes['field'], info_boxes['pos'], info_boxes['rpos'], info_boxes['value']
        for idx, (f, p, rp, v) in enumerate(zip(field, pos, rpos, value)):
            if idx < self.max_field:
                seq_info[idx, 0] = self.field2ind[f] if f in self.field2ind else self.UNK_FIELD
                seq_info[idx, 1] = p
                seq_info[idx, 2] = rp
                seq_info[idx, 3] = self.word2ind[v] if v in self.word2ind else self.UNK_TOKEN
            else:
                break
        return seq_info

    def get_data_loader(self, mode, batch_size, shuffle, device):
        if mode == 'train':
            seq_info_tensor = torch.tensor(
                self.seq_info_train, dtype=torch.long, device=device)
            seq_target_tensor = torch.tensor(
                self.seq_target_train, dtype=torch.long, device=device)
        elif mode == 'dev':
            seq_info_tensor = torch.tensor(
                self.seq_info_dev, dtype=torch.long, device=device)
            seq_target_tensor = torch.tensor(
                self.seq_target_dev, dtype=torch.long, device=device)
        else:
            raise ValueError('illegal mode: ', mode)
        dataset = Data.TensorDataset(seq_info_tensor, seq_target_tensor)
        data_loader = Data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
        return data_loader

    def get_refs(self, tag='test'):
        """ get gold summaries """
        if tag == 'dev':
            tag_data = self.dev_data
        elif tag == 'test':
            tag_data = self.test_data
        else:
            raise ValueError('invalid tag: ', tag)
        list_refs = []
        for data_item in tag_data:
            list_refs.append(data_item['text'])
        return list_refs

    def translate(self, list_seq):
        """ translate sequence-in-numbers to real sentence """
        list_token = []
        for index in list_seq:
            if index == self.UNK_TOKEN:
                continue
            elif index == self.SOS_TOKEN:
                continue
            elif index == self.PAD_TOKEN:
                continue
            elif index == self.EOS_TOKEN:
                break
            else:
                list_token.append(self.ind2word[str(int(index))])
        return ' '.join(list_token)

    def translate_w_copy(self, list_seq, attn_score, data_idx):
        """ translate sequence-in-numbers to real sentence with copy mechanism """
        list_token = []
        for index, attn in zip(list_seq, attn_score):
            if index == self.UNK_TOKEN:
                attn_max = attn.max(0)[1].item()
                if attn_max < len(self.test_data[data_idx]['info_box']['value']):
                    list_token.append(
                        self.test_data[data_idx]['info_box']['value'][attn_max])
                else:
                    list_token.append('<unk>')
            elif index == self.SOS_TOKEN:
                continue
            elif index == self.PAD_TOKEN:
                continue
            elif index == self.EOS_TOKEN:
                break
            else:
                list_token.append(self.ind2word[str(int(index))])

        return ' '.join(list_token)

    @staticmethod
    def _load_data(path, mode='r'):
        with open(path, mode=mode) as f:
            return json.load(f)
