import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from field_lstm import CustomLstmCell, BidirectionalLSTM
from beam_search import beam_search


class EncoderAttn(nn.Module):
    def __init__(self, field_vocab_size, pos_size, word_vocab_size,
                 embed_dim_field, embed_dim_pos, embed_dim_word, hidden_dim, dropout_p):
        super(EncoderAttn, self).__init__()
        self.embed_dim_field = embed_dim_field
        self.embed_dim_pos = embed_dim_pos
        self.embed_dim_word = embed_dim_word
        self.hidden_dim = hidden_dim

        self.field_embedding = nn.Embedding(field_vocab_size, embed_dim_field)
        self.pos_embedding = nn.Embedding(pos_size, embed_dim_pos)
        self.value_embedding = nn.Embedding(word_vocab_size, embed_dim_word)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bi_lstm = BidirectionalLSTM(CustomLstmCell, embed_dim_word + embed_dim_field, hidden_dim,
                                         2 * embed_dim_pos)
        self.fc_lstm = nn.Linear(hidden_dim * 2, hidden_dim)
        self.weight_reply = nn.Parameter(torch.randn(
            hidden_dim, hidden_dim), requires_grad=True)
        self.fc_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layernorm_gate = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.layernorm_lstm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def _content_selection_gate(self, reply_encoder):
        # reply_pre(batch, record_num, hid_dim), reply_post(batch, hid_dim, record_num)
        reply_pre = reply_encoder.permute(1, 0, 2)
        reply_post = reply_pre.permute(0, 2, 1)
        alpha = F.softmax(torch.matmul(
            torch.matmul(reply_pre, self.weight_reply), reply_post), dim=2)
        reply_c = torch.bmm(alpha, reply_pre)
        attn_gate = torch.sigmoid(self.layernorm_gate(
            self.fc_gate(torch.cat((reply_pre, reply_c), dim=2))))
        reply_cs = attn_gate * reply_pre
        return reply_cs

    def forward(self, encoder_input):
        field_embed = self.field_embedding(encoder_input[:, :, 0])
        pos_embed = self.pos_embedding(encoder_input[:, :, 1])
        rpos_embed = self.pos_embedding(encoder_input[:, :, 2])
        value_embed = self.value_embedding(encoder_input[:, :, 3])
        field_value = self.dropout(
            torch.cat((field_embed, value_embed), dim=2))
        bi_pos = torch.cat((pos_embed, rpos_embed), dim=2)

        encoder_output, (h_n, c_n) = self.bi_lstm(
            seq_input=field_value, field_pos=bi_pos)
        encoder_output = self.layernorm_lstm((self.fc_lstm(encoder_output)))
        h_n = self.layernorm_lstm(self.fc_lstm(
            torch.cat((h_n[-2], h_n[-1]), dim=1)))
        c_n = self.layernorm_lstm(self.fc_lstm(
            torch.cat((c_n[-2], c_n[-1]), dim=1)))
        content_selection = self._content_selection_gate(encoder_output)

        return content_selection, (h_n, c_n)


class DecoderAttn(nn.Module):
    def __init__(self, word_vocab_size, embed_dim, hidden_dim, dropout_p):
        super(DecoderAttn, self).__init__()
        self.output_dim = word_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(word_vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, word_vocab_size)

    @staticmethod
    def _weighted_encoder_rep(decoder_hidden, content_selection):
        energy = (decoder_hidden.unsqueeze(1) * content_selection) / \
            math.sqrt(decoder_hidden.size(-1))
        attn_score = F.softmax(torch.sum(energy, dim=2), dim=1)
        attn_with_selector = attn_score.unsqueeze(dim=2) * content_selection
        return torch.sum(attn_with_selector, dim=1), attn_score

    def forward(self, decoder_input, decoder_hidden, content_selection):
        embed = self.dropout(self.embedding(decoder_input))
        attn_vector, attn_score = self._weighted_encoder_rep(
            decoder_hidden[0], content_selection)
        emb_attn_combine = torch.cat((embed, attn_vector), dim=1)
        h_n, c_n = self.lstm(emb_attn_combine, decoder_hidden)
        decoder_output = F.log_softmax(self.fc_out(h_n), dim=1)

        return decoder_output, (h_n, c_n), attn_score


class Table2Text(nn.Module):
    def __init__(self, encoder, decoder, beam_width, max_len, max_field):
        super(Table2Text, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beam_width = beam_width
        self.max_len = max_len
        self.max_field = max_field

    def forward(self, seq_input, seq_target, train_mode):
        # encoder
        content_selection, decoder_hidden = self.encoder(seq_input)

        if train_mode:
            batch_size = seq_target.size(0)
            seq_output = torch.zeros(
                (batch_size, self.max_len, self.decoder.output_dim)).cuda()
            for timeStep in range(self.max_len):
                decoder_input = seq_target[:, timeStep]
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, content_selection)
                seq_output[:, timeStep, :] = decoder_output
            return seq_output
        else:
            if self.beam_width == 1:  # beam search with beam_width=1 equals to greedy search
                attn_map = torch.zeros((self.max_len, self.max_field)).cuda()
                seq_output = torch.zeros(self.max_len).cuda()
                decoder_input = seq_target  # first token: SOS_TOKEN
                for timeStep in range(self.max_len):
                    decoder_output, decoder_hidden, attn_score = self.decoder(
                        decoder_input, decoder_hidden, content_selection)
                    decoder_input = decoder_output.max(1)[1]
                    seq_output[timeStep] = decoder_input.squeeze()
                    attn_map[timeStep] = attn_score.squeeze()
                    if decoder_input.item() == 3:  # EOS_TOKEN
                        attn_map = attn_map[:timeStep]
                        break
            else:  # beam search
                seq_output, attn_map = beam_search(max_len=self.max_len,
                                                   max_field=self.max_field,
                                                   beam_width=self.beam_width,
                                                   decoder=self.decoder,
                                                   decoder_input=seq_target,
                                                   decoder_hidden=decoder_hidden,
                                                   content_selection=content_selection)
            return seq_output, attn_map


def detach(item):
    """ detach tensor from computational graph """
    if isinstance(item, tuple):
        return [hidden_state.detach() for hidden_state in item]
    elif isinstance(item, torch.Tensor):
        return item.detach()
    else:
        raise TypeError('invalid type: ', type(item))
