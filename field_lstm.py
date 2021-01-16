import torch
import torch.nn as nn
from torch.nn import Parameter, LayerNorm


class CustomLstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, field_size):
        super(CustomLstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.field_size = field_size

        self.weight_ih = Parameter(torch.randn(
            4 * hidden_size, input_size), requires_grad=True)
        self.weight_hh = Parameter(torch.randn(
            4 * hidden_size, hidden_size), requires_grad=True)
        self.weight_fh = Parameter(torch.randn(
            2 * hidden_size, field_size), requires_grad=True)

        # layer norm provides learn-able biases
        self.layernorm_i = LayerNorm(4 * hidden_size, elementwise_affine=True)
        self.layernorm_h = LayerNorm(4 * hidden_size, elementwise_affine=True)
        self.layernorm_f = LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.layernorm_c = LayerNorm(hidden_size, elementwise_affine=True)

    def forward(self, seq_input, states, field_pos):
        """ input_shape: (batch_size, input_size) """
        if states is None:
            h_x = torch.randn(seq_input.size(0), self.hidden_size,
                              requires_grad=False).cuda()
            c_x = torch.randn(seq_input.size(0), self.hidden_size,
                              requires_grad=False).cuda()
        else:
            h_x, c_x = states

        input_gates = self.layernorm_i(torch.mm(seq_input, self.weight_ih.t()))
        hidden_gates = self.layernorm_h(torch.mm(h_x, self.weight_hh.t()))
        gates = input_gates + hidden_gates
        forget_gate, update_gate, cell_gate, output_gate = torch.chunk(
            gates, chunks=4, dim=1)

        fd_pos = self.layernorm_f(torch.mm(field_pos, self.weight_fh.t()))
        fd, pos = torch.chunk(fd_pos, chunks=2, dim=1)

        forget_gate = torch.sigmoid(forget_gate)
        update_gate = torch.sigmoid(update_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        fd = torch.sigmoid(fd)
        pos = torch.tanh(pos)

        c_y = self.layernorm_c(forget_gate * c_x + update_gate * cell_gate +
                               fd * pos)
        h_y = output_gate * torch.tanh(c_y)

        return h_y, (h_y, c_y)


class LstmLayer(nn.Module):
    def __init__(self, cell, *args):
        """ *args: (input_size, hidden_size) """
        super(LstmLayer, self).__init__()
        self.cell = cell(*args)

    def forward(self, seq_input, states=None, field_pos=None):
        input_stack = seq_input.unbind(1)
        fp_stack = field_pos.unbind(1)
        outputs = []
        for i in range(len(input_stack)):
            out, states = self.cell(input_stack[i], states, fp_stack[i])
            outputs.append(out)
        return torch.stack(outputs), states


class ReverseLstmLayer(nn.Module):
    def __init__(self, cell, *args):
        """ *args: (input_size, hidden_size) """
        super(ReverseLstmLayer, self).__init__()
        self.cell = cell(*args)

    def forward(self, seq_input, states=None, field_pos=None):
        input_stack = reverse(seq_input.unbind(1))
        fp_stack = reverse(field_pos.unbind(1))
        outputs = []
        for i in range(len(input_stack)):
            out, states = self.cell(input_stack[i], states, fp_stack[i])
            outputs.append(out)
        return torch.stack(reverse(outputs)), states


class BidirectionalLSTM(nn.Module):
    def __init__(self, cell, *args):
        super(BidirectionalLSTM, self).__init__()
        self.layers = nn.ModuleList(
            [LstmLayer(cell, *args),
             ReverseLstmLayer(cell, *args)])

    def forward(self, seq_input, states=None, field_pos=None):
        outputs = []
        h_ns, c_ns = [], []
        for layer in self.layers:
            out, (h_n, c_n) = layer(seq_input=seq_input, field_pos=field_pos)
            outputs.append(out)
            h_ns.append(h_n.unsqueeze(dim=0))
            c_ns.append(c_n.unsqueeze(dim=0))

        outputs = torch.cat(outputs, dim=-1)
        h_n = torch.cat(h_ns, dim=0)
        c_n = torch.cat(c_ns, dim=0)
        return outputs, (h_n, c_n)


def reverse(tensor_list):
    """ reverse List[Tensor] """
    return tensor_list[::-1]
