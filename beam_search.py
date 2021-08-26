from queue import PriorityQueue
import torch


class BeamSearchNode(object):
    def __init__(self, previous_hidden, previous_attn, previous_node,
                 current_input, inherited_order, prob, length):
        super().__init__()

        self.previous_hidden = previous_hidden
        self.previous_attn = previous_attn
        self.previousNode = previous_node
        self.current_input = current_input
        self.inherited_order = inherited_order
        self.prob = prob
        self.length = length

    def __eq__(self, other):
        return self.inherited_order == other.inherited_order

    def __lt__(self, other):
        return self.inherited_order < other.inherited_order


def beam_search(max_len, max_field, beam_width,
                decoder, decoder_input, decoder_hidden, content_selection):
    EOS_TOKEN = 3
    nodes = PriorityQueue()
    next_nodes = PriorityQueue()

    node = BeamSearchNode(previous_hidden=decoder_hidden,
                          previous_attn=None,
                          previous_node=None,
                          current_input=decoder_input,
                          inherited_order=1,
                          prob=0.0,
                          length=1)
    nodes.put((node.prob, node))
    while True:
        # pop one then put it back or break loop, to check if has arrive EOS_TOKEN or max_length
        eos_prob, eos_node = nodes.get()
        if eos_node.length >= max_len:
            end_node = eos_node
            break
        if eos_node.current_input.item() == EOS_TOKEN and eos_node.previousNode is not None:
            end_node = eos_node
            break
        nodes.put((eos_prob, eos_node))

        qsize = nodes.qsize()
        for idx in range(qsize):
            _, node = nodes.get()
            previous_hidden = node.previous_hidden
            current_input = node.current_input
            decoder_output, decoder_hidden, attn_score = decoder(decoder_input=current_input,
                                                                 decoder_hidden=previous_hidden,
                                                                 content_selection=content_selection)
            values, index = decoder_output.data.topk(beam_width)
            for beam in range(beam_width):
                beam_out = index[0, beam].unsqueeze(dim=0)
                # decoder output probabilities with log_softmax
                beam_prob = -values[0, beam]
                next_node = BeamSearchNode(previous_hidden=decoder_hidden,
                                           previous_attn=attn_score.squeeze(),
                                           previous_node=node,
                                           current_input=beam_out,
                                           inherited_order=idx,
                                           prob=node.prob + beam_prob,
                                           length=node.length + 1)
                next_nodes.put((next_node.prob, next_node))
        for beam in range(beam_width):
            # most probable [beam_width]_num nodes
            temp_prob, temp_node = next_nodes.get()
            nodes.put((temp_prob, temp_node))
        next_nodes.queue.clear()  # clear next_nodes

    beam_search_out = end_node.current_input
    attn_list = [end_node.previous_attn]
    while end_node.previousNode is not None:
        end_node = end_node.previousNode
        beam_search_out = torch.cat(
            (end_node.current_input, beam_search_out), dim=0)
        if end_node.previous_attn is not None:
            attn_list.append(end_node.previous_attn)
        else:
            attn_list.append(torch.zeros(max_field).cuda())
    attn_list.reverse()
    beam_search_attn = torch.stack(tuple(attn_list), dim=0)

    return beam_search_out[1:], beam_search_attn[1:]
