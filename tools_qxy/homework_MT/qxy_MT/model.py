"""
    本代码用于: 用于定义「Transformer 模型结构」和「Beam_search」
    创建时间: 2021 年 12 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 模型创建相关的 ----- #
import torch
import torch.nn.functional as F  
import torch.nn as nn
from torch.autograd import Variable

# ----- 数据处理相关的 ----- #
import math
import copy

# ----- 导入自定义的包 ----- #
from dataset import subsequent_mask

# ==================== 函数实现 ==================== #
# ---------- Transformer 模型的定义 ----- #
def transformer_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    # ----- 多头自注意力层 ----- #
    attn = MultiHeadedAttention(h, d_model)

    # ----- FFN 层 ----- #
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # ----- 位置嵌入 ----- #
    position = PositionalEncoding(d_model, dropout)

    # ----- 定义 Transformer 模型 ----- #
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # ----- 参数初始化 ----- #
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  
    return model


# ----- Embedding 层的实现 ----- #
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ----- 位置嵌入的实现 ----- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad = False)
        self.position = torch.arange(0., max_len).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)

        self.pe.data = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# ----- 一个注意力层的实现 ----- #
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 进行 Mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

# ----- 多头注意力层的实现 ----- #
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ----- 层正则化的实现 ----- # 
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


# ----- 一层 layer 的实现 ----- #
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ----- FFN 的实现 ----- #
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ----- 定义 Encoder ----- #
class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ----- 定义一层 Encoder ----- #
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ----- 定义 Decoder ----- #
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        # ----- 复制 N 个 encoder layer ----- #
        self.layers = clones(layer, N)
        
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ----- 定义一层 Decoder ----- #
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# ----- 生成部分的定义 ----- #
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# ----- 定义整个 Transformer 结构 ----- #
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        decode_out = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        return self.generator(decode_out)


# ---------- Beam Search 的实现 ---------- #
""" Beam search """
class Beam:
    def __init__(self, size, pad, bos, eos, device=False):
        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        # Initialize to [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        """Update beam status and check if finished or not."""
        num_words = word_logprob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # in initial case,
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        # print(k.type())
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    """ Translation work in one batch """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ Indicate the position of an instance in a tensor. """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """ Collect tensor parts associated to active instances. """

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # active instances (elements of batch) * beam search size x seq_len x h_dimension
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # select only parts of tensor which are still active
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(
            src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        """ Decode and update beam status, and then return active beam idx """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # Batch size x Beam size x Dec Seq Len
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # Batch size*Beam size x Dec Seq Len
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            out = model.decode(enc_output, src_mask,
                               dec_seq,
                               subsequent_mask(dec_seq.size(1))
                               .type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])  # Fill Beam object with assigned probabilities
                if not is_inst_complete:  # if top beam ended with eos, we do not add it
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # get decoding sequence for each beam
        # size: Batch size*Beam size x Dec Seq Len
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # get word probabilities for each beam
        # size: Batch size x Beam size x Vocabulary
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # -- Encode
        src_enc = model.encode(src, src_mask)

        #  Repeat data for beam search
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        # -- Prepare beams
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_len + 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            # filter out inactive tensor parts (for already decoded sequences)
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)

    return batch_hyp, batch_scores