import datetime
import time
import shutil
from tqdm import tqdm
from torch import optim
import torch.nn as nn

from utils import *
from parameters import parameters
from data_processor import DataProcessor
from evals import BleuScore, RougeScore
from early_stopping import EarlyStopping
from model import EncoderAttn, DecoderAttn, Table2Text

batch_size = parameters['batch_size']
max_epoch = parameters['max_epoch']
learning_rate = parameters['learning_rate']
field_emb_dim = parameters['field_emb_dim']
pos_embed_dim = parameters['pos_embed_dim']
word_embed_dim = parameters['word_embed_dim']
hidden_dim = parameters['hidden_dim']
dropout = parameters['dropout']
weight_decay = parameters['weight_decay']
grad_clip = parameters['grad_clip']
seed = parameters['seed']
beam_width = parameters['beam_width']
max_len = parameters['max_len']
max_field = parameters['max_field']
# report = params['report']
resume_train = parameters['resume']
copy = parameters['copy']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fix_seed(seed)

cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
logger = get_logger('./results/logs/' + cur_time + '.log')
logger.info(f'GPU_device: {torch.cuda.get_device_name()}')
for item in parameters:
    logger.info(f'{item}: {parameters[item]}')

start_time = time.time()
data_processor = DataProcessor()
train_data_loader = data_processor.get_data_loader(
    mode='train', batch_size=batch_size, shuffle=True, device=device)
dev_data_loader = data_processor.get_data_loader(
    mode='dev', batch_size=batch_size, shuffle=False, device=device)
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
                      field_emb_dim, pos_embed_dim, word_embed_dim, hidden_dim, dropout)
decoder = DecoderAttn(word_vocab_size, word_embed_dim, hidden_dim, dropout)
model = Table2Text(encoder, decoder, beam_width).to(device)
model.apply(weights_init)

optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.NLLLoss()
early_stop = EarlyStopping(mode='min', min_delta=0.001, patience=5)

if resume_train:
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
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
for epoch in range(1, int(max_epoch + 1)):
    start_time = time.time()
    train_loss = train(train_data_loader)
    dev_loss = validate(dev_data_loader)
    loss_dict_train.append(train_loss)
    loss_dict_dev.append(dev_loss)

    epoch_min, epoch_sec = record_time(start_time, time.time())
    logger.info(
        f'epoch: [{epoch:02}/{max_epoch}]  train_loss={train_loss:.3f}  valid_loss={dev_loss:.3f}  '
        f'duration: {epoch_min}m {epoch_sec}s')

    if early_stop.step(dev_loss):
        logger.info(f'early stop at [{epoch:02}/{max_epoch}]')
        break

if max_epoch > 0:
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
    seq_input = torch.tensor(data_processor.process_one_data(idx_data=idx_data), dtype=torch.long,
                             device=device).unsqueeze(0)
    seq_output, attn_map = evaluate(seq_input)
    list_seq = seq_output.squeeze().tolist()
    if not copy:
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
