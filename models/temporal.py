# coding：utf-8
# @Time ：2022/11/9 9:49
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder,TransformerDecoder
import math

class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, slide_win,max_len=1024):
        super(BiasedPositionalEmbedding, self).__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)
        self.Wt = nn.Parameter(torch.FloatTensor(slide_win,d_model))

    def forward(self, time):
        diff = torch.diff(time, dim=0)
        pad = torch.zeros(1,diff.shape[1],diff.shape[2]).float().to(time.device)
        diff = torch.cat([pad, diff], dim=0)
        interval = torch.round(diff / 0.0001736)

        phi = torch.einsum('wbi,wi->wbi', interval, self.Wt)
        arc = (self.position * self.div_term).unsqueeze(1)
        pe = torch.sin(arc + phi) + torch.cos(arc + phi)

        return pe,interval
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None,src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class Temporal(nn.Module):
    def __init__(self, embed_time,slide_win,small_win,dim):
        super(Temporal, self).__init__()
        self.small_win = small_win
        self.time_bias_encoder = BiasedPositionalEmbedding(embed_time*dim, slide_win,slide_win)

        self.Linear1 = nn.Linear(dim, embed_time*dim)
        self.Linear2 = nn.Linear(dim, embed_time*dim)

        self.encoder_layers = TransformerEncoderLayer(d_model=embed_time*dim, nhead=embed_time*dim // 2, dim_feedforward=16)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, 1)
        self.decoder_layers = TransformerDecoderLayer(d_model=embed_time*dim, nhead=embed_time*dim // 2, dim_feedforward=16)
        self.transformer_decoder = TransformerDecoder(self.decoder_layers, 1)

        self.Linear3 = nn.Linear(embed_time*dim,dim)
        self.fcn = nn.Sigmoid()
        

    def forward(self, src, tgt, src_time):
        src = self.Linear1(src)
        src_time_enc, inter = self.time_bias_encoder(src_time)
        src = src + src_time_enc
        tgt = self.Linear2(tgt)
        tgt_time_enc = src_time_enc[-self.small_win:, :, :]
        tgt = tgt + tgt_time_enc
        memory = self.transformer_encoder(src).float()
        x = self.transformer_decoder(tgt.float(), memory)
        x = self.Linear3(x)
        x = self.fcn(x).float()
        return x,memory