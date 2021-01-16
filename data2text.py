import time
import datetime
from parameters import parameters
from data_processor import DataProcessor
from model import EncoderAttn, DecoderAttn, Table2Text
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_text(test_input, net, copy, test_idx):
    net.eval()
    sos_target = torch.tensor([2], dtype=torch.long, device=device)
    with torch.no_grad():
        test_output, attn = net(test_input, sos_target, train_mode=False)
    if not copy:
        text_gen = data_processor.translate(test_output.squeeze().tolist())
    else:
        text_gen = data_processor.translate_w_copy(list_seq=test_output.squeeze().tolist(),
                                                   attn_score=attn,
                                                   data_idx=test_idx)
    return text_gen, attn


if __name__ == '__main__':

    field_emb_dim = parameters['field_emb_dim']
    pos_embed_dim = parameters['pos_embed_dim']
    word_embed_dim = parameters['word_embed_dim']
    hidden_dim = parameters['hidden_dim']
    dropout = parameters['dropout']
    beam_width = parameters['beam_width']
    if_copy = parameters['copy']

    cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    data_processor = DataProcessor()
    print('finish processing data ...')
    field_vocab_size = data_processor.field_vocab_size
    pos_size = data_processor.pos_size
    word_vocab_size = data_processor.word_vocab_size

    encoder = EncoderAttn(field_vocab_size, pos_size, word_vocab_size,
                          field_emb_dim, pos_embed_dim, word_embed_dim, hidden_dim, dropout)
    decoder = DecoderAttn(word_vocab_size, word_embed_dim, hidden_dim, dropout)
    model = Table2Text(encoder, decoder, beam_width=beam_width).to(device)

    # load the latest checkpoint
    checkpoint, cp_name = load_checkpoint(latest=True)
    model.load_state_dict(checkpoint['model'])
    print(f'finish loading model: [{cp_name}]')

    # choose 10 utterances randomly to check
    rand_list = random_list(0, len(data_processor.test_data), 10)
    print(f'random_list: {rand_list}')
    check_file_exist('./results/utterances')
    start = time.time()

    with open('./results/utterances/' + cur_time + '.txt', 'w') as f:
        f.write(cur_time + '\n')

        for idx, idx_data in enumerate(rand_list):
            seq_input = torch.tensor(data_processor.process_one_data(
                idx_data=idx_data), dtype=torch.long, device=device).unsqueeze(dim=0)
            text, attn_map = generate_text(seq_input, model, if_copy, idx_data)

            f.write('                    system \n')
            f.write(text + '\n')
            f.write('                     gold \n')
            f.write(data_processor.test_data[idx_data]['text'].strip() + '\n')
            f.write('################################################## \n\n')

            save_attention_map(attn_map=attn_map, fields=data_processor.test_data[idx_data]['info_box']['value'],
                               text=text.split(' '), img_name=str(cur_time) + '_' + str(idx + 1))

    duration = time.time() - start
    print(f'average duration: {(duration / 10):.2f}s')
