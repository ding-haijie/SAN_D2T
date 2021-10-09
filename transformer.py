import math
import argparse
import datetime
import time
import shutil
from tqdm import tqdm

import torch.nn as nn
from torch import optim

from utils import *
from data_processor import DataProcessor
from evals import BleuScore, RougeScore
from early_stopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# hyper params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Mini batch size')
parser.add_argument('--max_epoch', default=30, type=int,
                    help='Max epoch for training.')
parser.add_argument('--max_len', default=60, type=int,
                    help='Max length of the texts.')
parser.add_argument('--max_field', default=100, type=int,
                    help='Max length of the fields.')
# model params
parser.add_argument('--field_emb_dim', default=50, type=int,
                    help='Dimension of field embedding.')
parser.add_argument('--word_emb_dim', default=400, type=int,
                    help='Dimension of word embedding.')
parser.add_argument('--hidden_dim', default=512, type=int,
                    help='Dimension of hidden layer.')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate.')
parser.add_argument('--n_head', default=8, type=int,
                    help='Number of heads in the multiheadattention models')
parser.add_argument('--num_encoder_layers', default=6, type=int,
                    help='Number of sub-encoder-layers')
parser.add_argument('--num_decoder_layers', default=6, type=int,
                    help='Number of sub-decoder-layers')
# optimizer params
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='Weight decay (L2 penalty).')
# others
parser.add_argument('--random_seed', default=1, type=int,
                    help='Seed for generating random numbers.')
parser.add_argument('--train_mode', default=True, type=bool,
                    help='If False, then doing inference.')
parser.add_argument('--resume', default=False, type=bool,
                    help='Whether to load checkpoints to resume training.')
args = parser.parse_args()
fix_seed(args.random_seed)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout_p, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        # compute position encodings in log space
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        pos_embedding = torch.zeros(max_len, embed_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(1)
        # add buffer[pos_embedding] to the module
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:x.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, field_vocab_size, word_vocab_size,
                 embed_dim_field, embed_dim_word, dropout_p,
                 n_head, num_encoder_layers, num_decoder_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout_p)

        self.field_embedding = nn.Embedding(field_vocab_size, embed_dim_field)
        self.value_embedding = nn.Embedding(word_vocab_size, embed_dim_word)
        self.target_embedding = nn.Embedding(word_vocab_size, hidden_dim)
        self.fc_field_value = nn.Linear(
            embed_dim_field + embed_dim_word, hidden_dim)

        self.transformer = nn.Transformer(d_model=hidden_dim,
                                          nhead=n_head,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=hidden_dim,
                                          dropout=dropout_p)
        self.fc_out = nn.Linear(hidden_dim, word_vocab_size)

    def forward(self, seq_input, seq_target,
                src_mask, target_mask,
                src_padding_mask, target_padding_mask, memory_padding_mask):
        field_embed = self.field_embedding(seq_input[:, :, 0])
        value_embed = self.value_embedding(seq_input[:, :, 3])
        field_value = self.dropout(self.fc_field_value(
            torch.cat((field_embed, value_embed), dim=2)))
        src = self.pos_encoding(field_value)
        target = self.pos_encoding(self.target_embedding(seq_target))

        seq_output = self.transformer(src=src, tgt=target,
                                      src_mask=src_mask, tgt_mask=target_mask,
                                      src_key_padding_mask=src_padding_mask,
                                      tgt_key_padding_mask=target_padding_mask,
                                      memory_key_padding_mask=memory_padding_mask)
        return self.fc_out(seq_output)

    def encode(self, seq_input, src_mask):
        field_embed = self.field_embedding(seq_input[:, :, 0])
        value_embed = self.value_embedding(seq_input[:, :, 3])
        field_value = self.fc_field_value(
            torch.cat((field_embed, value_embed), dim=2))
        src = self.pos_encoding(field_value)
        return self.transformer.encoder(src, src_mask)

    def decode(self, seq_target, memory, target_mask):
        target = self.pos_encoding(self.target_embedding(seq_target))
        return self.transformer.decoder(target, memory, target_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(seq_input, seq_target):
    # prevent model to look into the future words when making predictions
    src_len = seq_input.shape[0]
    target_len = seq_target.shape[0]

    src_mask = torch.zeros((src_len, src_len), device=device).to(torch.bool)
    target_mask = generate_square_subsequent_mask(target_len)

    src_padding_mask = (seq_input == 0).transpose(0, 1)  # PAD
    target_padding_mask = (seq_target == 0).transpose(0, 1)  # PAD
    return src_mask, target_mask, src_padding_mask, target_padding_mask


cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
logger = get_logger('./results/logs/' + cur_time + '.log')

# print arguments
for arg in vars(args):
    logger.info("{} : {}".format(arg, getattr(args, arg)))

# process data
start_time = time.time()
data_processor = DataProcessor(args)
train_data_loader = data_processor.get_data_loader(
    mode='train', batch_size=args.batch_size, shuffle=True, device=device)
dev_data_loader = data_processor.get_data_loader(
    mode='dev', batch_size=args.batch_size, shuffle=False, device=device)
field_vocab_size = data_processor.field_vocab_size
word_vocab_size = data_processor.word_vocab_size
logger.info(f'data processing consumes: {(time.time() - start_time):.2f}s')

# define the transformer model
model = TransformerModel(field_vocab_size, word_vocab_size,
                         args.field_emb_dim, args.word_emb_dim, args.dropout,
                         args.n_head, args.num_encoder_layers, args.num_decoder_layers,
                         args.hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()  # PAD
early_stop = EarlyStopping(mode='min', min_delta=0.001, patience=5)

if args.resume:
    checkpoint, cp_name = load_checkpoint(latest=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info(f'load checkpoint: [{cp_name}]')


def train(data_loader):
    model.train()
    epoch_loss = 0
    for train_input, train_target in tqdm(data_loader):
        train_input = train_input.permute(1, 0, 2)
        train_target = train_target.permute(1, 0)
        optimizer.zero_grad()
        src_mask, target_mask, src_padding_mask, target_padding_mask = create_mask(
            train_input[:, :, 3], train_target)
        train_output = model(train_input, train_target, src_mask, target_mask,
                             src_padding_mask, target_padding_mask, src_padding_mask)
        train_output = train_output[:, :-1].reshape(-1, train_output.size(-1))
        loss = criterion(train_output, train_target[:, 1:].reshape(-1))
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    return epoch_loss / len(data_loader)


def validate(data_loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for dev_input, dev_target in data_loader:
            dev_input = dev_input.permute(1, 0, 2)
            dev_target = dev_target.permute(1, 0)
            src_mask, target_mask, src_padding_mask, target_padding_mask = create_mask(
                dev_input[:, :, 3], dev_target)
            dev_output = model(dev_input, dev_target, src_mask, target_mask,
                               src_padding_mask, target_padding_mask, src_padding_mask)
            dev_output = dev_output[:, :-1].reshape(-1, dev_output.size(-1))
            loss = criterion(dev_output, dev_target[:, 1:].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def evaluate(infer_input):
    model.eval()
    with torch.no_grad():
        seq_len = infer_input.size(0)
        src_mask = (torch.zeros(seq_len, seq_len)).to(torch.bool).to(device)
        memory = model.encode(infer_input, src_mask)
        infer_target = torch.ones(1, 1).fill_(2).to(
            torch.long).to(device)  # SOS_TOKEN
        for _ in range(args.max_len):
            target_mask = generate_square_subsequent_mask(
                infer_target.size(0)).to(torch.bool)
            decoder_output = model.decode(infer_target, memory, target_mask)
            decoder_output = decoder_output.transpose(
                0, 1)  # shape(1, t, vocab_size)
            prob = model.fc_out(decoder_output[:, -1])
            next_word = prob.max(dim=1)[1].item()
            infer_target = torch.cat([infer_target, torch.ones(
                1, 1).type_as(infer_input.data).fill_(next_word).to(device)], dim=0)
            if next_word == 3:  # EOS_TOKEN
                break
        seq_output = infer_target.flatten()
    return seq_output


# define training and validating loop
loss_dict_train, loss_dict_dev = [], []
for epoch in range(1, int(args.max_epoch + 1)):
    start_time = time.time()
    train_loss = train(train_data_loader)
    dev_loss = validate(dev_data_loader)
    loss_dict_train.append(train_loss)
    loss_dict_dev.append(dev_loss)

    epoch_min, epoch_sec = record_time(start_time, time.time())
    logger.info(
        f'epoch: [{epoch:02}/{args.max_epoch}]  train_loss={train_loss:.3f}  valid_loss={dev_loss:.3f}  '
        f'duration: {epoch_min}m {epoch_sec}s')

    if early_stop.step(dev_loss):
        logger.info(f'early stop at [{epoch:02}/{args.max_epoch}]')
        break

if args.max_epoch > 0:
    save_checkpoint(experiment_time=cur_time, model=model, optimizer=optimizer)

# evaluate
check_file_exist('./results/rouge/system')
check_file_exist('./results/rouge/gold')
bleu_scorer = BleuScore()
rouge_scorer = RougeScore(system_dir='./results/rouge/system',
                          model_dir='./results/rouge/gold',
                          n_gram=4)
ref_summaries = data_processor.get_refs(tag='test')
bleu_scorer.set_refs(ref_summaries)
rouge_scorer.set_refs(ref_summaries)

for idx_data in range(len(data_processor.test_data)):
    seq_input = torch.tensor(data_processor.process_one_data(
        idx_data=idx_data), dtype=torch.long, device=device).unsqueeze(0)
    seq_output = evaluate(seq_input)
    list_seq = seq_output.squeeze().tolist()
    text_gen = data_processor.translate(list_seq)

    bleu_scorer.add_gen(text_gen)
    rouge_scorer.add_gen(text_gen)

bleu_score = bleu_scorer.calculate()
logger.info(f'bleu score: {bleu_score:.2f}')

rouge_scorer.file_writer()
rouge_score_dict = rouge_scorer.calculate()
for n_gram in rouge_score_dict:
    logger.info(f'{n_gram}: {rouge_score_dict[n_gram]:.2f}')

clean_logs(latest=True)  # clean chaotic logs caused by pyrouge

# remove redundant folders
shutil.rmtree('./results/rouge/system')
shutil.rmtree('./results/rouge/gold')
