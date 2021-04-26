import argparse
import datetime
import time
import shutil
from tqdm import tqdm
from torch import optim
import torch.nn as nn

from utils import *
from data_processor import DataProcessor
from evals import BleuScore, RougeScore
from early_stopping import EarlyStopping
from model import EncoderAttn, DecoderAttn, Table2Text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=32,
                    type=int, help='Mini batch size')
parser.add_argument('--max_epoch', default=40, type=int,
                    help='Max epoch for training.')
parser.add_argument('--lr', default=3e-4, type=float,
                    help='Initial learning rate.')
parser.add_argument('--field_emb_dim', default=50, type=int,
                    help='Dimension of field embedding.')
parser.add_argument('--pos_emb_dim', default=5, type=int,
                    help='Dimension of position embedding.')
parser.add_argument('--word_emb_dim', default=400, type=int,
                    help='Dimension of word embedding.')
parser.add_argument('--hidden_dim', default=500, type=int,
                    help='Dimension of hidden layer.')
parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate.')
parser.add_argument('--weight_decay', default=0.0,
                    type=float, help='Weight decay (L2 penalty).')
parser.add_argument('--grad_clip', default=5.0, type=float,
                    help='Max norm of the gradients clipping.')
parser.add_argument('--random_seed', default=1, type=int,
                    help='Seed for generating random numbers.')
parser.add_argument('--beam_width', default=1, type=int,
                    help='Size of beam search width.')
parser.add_argument('--max_len', default=60, type=int,
                    help='Max length of the texts.')
parser.add_argument('--max_field', default=100, type=int,
                    help='Max length of the fields.')
parser.add_argument('--pos_size', default=31, type=int,
                    help='Max number of position.')
parser.add_argument('--train', default=True, type=bool,
                    help='If False, then doing inference.')
parser.add_argument('--resume', default=False, type=bool,
                    help='Whether to load checkpoints to resume training.')
parser.add_argument('--copy', default=True, type=bool,
                    help='Whether to use copy mechanism.')
args = parser.parse_args()

fix_seed(args.random_seed)

cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
logger = get_logger('./results/logs/' + cur_time + '.log')

for arg in vars(args):  # print arguments
    logger.info("{} : {}".format(arg, getattr(args, arg)))

start_time = time.time()
data_processor = DataProcessor(args)
train_data_loader = data_processor.get_data_loader(
    mode='train', batch_size=args.batch_size, shuffle=True)
dev_data_loader = data_processor.get_data_loader(
    mode='dev', batch_size=args.batch_size, shuffle=False)
field_vocab_size = data_processor.field_vocab_size
pos_size = data_processor.pos_size
word_vocab_size = data_processor.word_vocab_size

logger.info(f'data processing consumes: {(time.time() - start_time):.2f}s')


def weights_init(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0.0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


encoder = EncoderAttn(field_vocab_size, pos_size, word_vocab_size,
                      args.field_emb_dim, args.pos_emb_dim, args.word_emb_dim,
                      args.hidden_dim, args.dropout)
decoder = DecoderAttn(word_vocab_size,
                      args.word_embed_dim, args.hidden_dim, args.dropout)

model = Table2Text(encoder, decoder, args.beam_width,
                   args.max_len, args.max_field).to(device)

optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.NLLLoss()
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
        optimizer.zero_grad()
        train_output = model(train_input, train_target, train_mode=True)
        train_output = train_output[:, :-1].reshape(-1, train_output.size(-1))
        loss = criterion(train_output, train_target[:, 1:].reshape(-1))
        loss.backward()
        epoch_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    return epoch_loss / len(data_loader)


def validate(data_loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for validate_input, validate_target in data_loader:
            validate_output = model(
                validate_input, validate_target, train_mode=True)
            validate_output = validate_output[:, :-
                                              1].reshape(-1, validate_output.size(-1))
            loss = criterion(
                validate_output, validate_target[:, 1:].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def evaluate(infer_input):
    model.eval()
    with torch.no_grad():
        infer_target = torch.tensor(
            [2], dtype=torch.long, device=device)  # SOS_TOKEN
        eval_output, attn = model(infer_input, infer_target, train_mode=False)
    return eval_output, attn


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
    seq_output, attn_map = evaluate(seq_input)
    list_seq = seq_output.squeeze().tolist()
    if not args.copy:
        text_gen = data_processor.translate(list_seq)
    else:
        text_gen = data_processor.translate_w_copy(
            list_seq=list_seq, attn_score=attn_map, data_idx=idx_data)

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
