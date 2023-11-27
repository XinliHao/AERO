import torch.nn as nn
from parser import *
from models.temporal import Temporal
from models.concurrent import Concurrent
from ablation.transshort import TransShort
from ablation.complateGraph import ComplateGraph
class OnlyTemporal(nn.Module):
    def __init__(self, dim, embed_time,slide_win,small_win):
        super(OnlyTemporal, self).__init__()
        self.name = 'OnlyTemporal'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time,slide_win,small_win,1)
    
    def forward(self, inputW):
        recon, memory_list = [], []
        # 遍历每一个维度，将每一个维度的数据单独输入
        for i in range(1, inputW.shape[-1]):
            # input:[128,100,1],  embedding[128,100,1]
            # src[50,128,1] tgt[10,128,1]
            input_trans = inputW.permute(1, 0, 2)
        
            src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
            src = input_trans[:, :, i].view(input_trans.shape[0], input_trans.shape[1], 1)
        
            tgt_time = input_trans[-self.small_win:, :, 0].view(self.small_win, input_trans.shape[1], 1)
            tgt = input_trans[-self.small_win:, :, i].view(self.small_win, input_trans.shape[1], 1)
        
            trans, memory = self.trans(src, tgt, src_time, tgt_time)
            recon.append(trans)
            memory_list.append(memory)
    
        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        return recon1


class OnlyTemporalMulti(nn.Module):
    def __init__(self, dim, embed_time, slide_win, small_win):
        super(OnlyTemporalMulti, self).__init__()
        self.name = 'OnlyTemporalMulti'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time, slide_win, small_win,dim)
    
    def forward(self, inputW):
        input_trans = inputW.permute(1, 0, 2)
        
        src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
        src = input_trans[:, :,1:].view(input_trans.shape[0], input_trans.shape[1], -1)
        
        tgt_time = input_trans[-self.small_win:, :, 0].view(self.small_win, input_trans.shape[1], 1)
        tgt = input_trans[-self.small_win:, :, 1:].view(self.small_win, input_trans.shape[1], -1)
        
        recon, memory = self.trans(src, tgt, src_time, tgt_time)
        recon1 = recon.permute(1, 0, 2)
        return recon1


class OnlyConcurrent(nn.Module):
    def __init__(self, dim,slide_win,small_win):
        super(OnlyConcurrent, self).__init__()
        self.name = 'OnlyConcurrent'
        self.slide_win = slide_win
        self.small_win = small_win
        self.reslayer = Concurrent(dim, small_win, small_win, small_win)
    
    def forward(self, dataW):
        inputW = dataW[:, -self.small_win:, 1:]
        imput_RGCN = inputW.permute(0, 2, 1)
        recon2 = self.reslayer(imput_RGCN, imput_RGCN)
        return recon2


class StaticGraph(nn.Module):
    def __init__(self, dim, embed_time,slide_win,small_win):
        super(StaticGraph, self).__init__()
        self.name = 'StaticGraph'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time, slide_win, small_win, 1)
        # GDN
        self.reslayer = ComplateGraph(dim, small_win, small_win, small_win)
    
    def forward(self, inputW, stage):
        if self.train:
            if stage == 1:
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = False
                for name, param in self.trans.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.trans.named_parameters():
                    param.requires_grad = False
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = True
        

        recon, res, origin, memory_list = [], [], [], []

        for i in range(1, inputW.shape[-1]):
            input_trans = inputW.permute(1, 0, 2)
            
            src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
            src = input_trans[:, :, i].view(input_trans.shape[0], input_trans.shape[1], 1)
            
            tgt_time = input_trans[-self.small_win:, :, 0].view(self.small_win, input_trans.shape[1], 1)
            tgt = input_trans[-self.small_win:, :, i].view(self.small_win, input_trans.shape[1], 1)
            
            trans, memory = self.trans(src, tgt, src_time, tgt_time)
            recon.append(trans)
            origin.append(tgt)
            res.append(tgt - trans)
            memory_list.append(memory)
            
        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        origin_full = torch.cat(origin, 2).permute(1, 2, 0)
        res_full = torch.cat(res, 2).permute(1, 2, 0)
        recon2 = self.reslayer(res_full, origin_full)
        return recon1, recon2
    
class MultiVariate(nn.Module):
    def __init__(self, dim, embed_time, slide_win, small_win):
        super(MultiVariate, self).__init__()
        self.name = 'MultiVariate'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time, slide_win, small_win, dim)
        self.reslayer = Concurrent(dim, small_win, small_win, small_win)
    
    def forward(self, inputW, stage):
        if self.train:
            if stage == 1:
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = False
                for name, param in self.trans.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.trans.named_parameters():
                    param.requires_grad = False
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = True
        
        input_trans = inputW.permute(1, 0, 2)
        
        src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
        src = input_trans[:, :, 1:].view(input_trans.shape[0], input_trans.shape[1], -1)
        
        tgt_time = input_trans[-self.small_win:, :, 0].view(self.small_win, input_trans.shape[1], 1)
        tgt = input_trans[-self.small_win:, :, 1:].view(self.small_win, input_trans.shape[1], -1)
        
        recon, memory = self.trans(src, tgt, src_time, tgt_time)
        res = tgt - recon
        
        recon1 = recon.permute(1, 0, 2)
        origin_full = tgt.permute(1, 2, 0)
        res_full = res.permute(1, 2, 0)
        recon2 = self.reslayer(res_full, origin_full)
        return recon1, recon2


from ablation.esg import ESG
class DynamicGraph(nn.Module):
    def __init__(self, embed_time, slide_win, small_win, fc_dim, node_fea):
        super(DynamicGraph, self).__init__()
        self.name = 'DynamicGraph'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time, slide_win, small_win, 1)  # 这个1表示，每一个维度单独进入
        self.reslayer = ESG(dy_embedding_dim=20, dy_interval=[31, 31, 21, 14, 1], num_nodes=137, seq_length=small_win,
                            pred_len=1,
                            in_dim=1, out_dim=1, n_blocks=1, n_layers=1, conv_channels=16, residual_channels=16,
                            skip_channels=32,
                            end_channels=64, kernel_set=[2, 3, 6, 7], dilation_exp=2, gcn_depth=2,
                            fc_dim=fc_dim, st_embedding_dim=40, dropout=0.3, propalpha=0.05, layer_norm_affline=False,
                            static_feat=node_fea)
    
    def forward(self, inputW, stage):
        if self.train:
            if stage == 1:
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = False
                for name, param in self.trans.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.trans.named_parameters():
                    param.requires_grad = False
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = True
        
        # dataW:[128,100,21]
        recon, res, origin, memory_list = [], [], [], []
        # 遍历每一个维度，将每一个维度的数据单独输入
        for i in range(1, inputW.shape[-1]):
            input_trans = inputW.permute(1, 0, 2)
            
            src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
            src = input_trans[:, :, i].view(input_trans.shape[0], input_trans.shape[1], 1)
            
            tgt_time = input_trans[-self.small_win:, :, 0].view(self.small_win, input_trans.shape[1], 1)
            tgt = input_trans[-self.small_win:, :, i].view(self.small_win, input_trans.shape[1], 1)
            
            trans, memory = self.trans(src, tgt, src_time, tgt_time)
            recon.append(trans)
            origin.append(tgt)
            res.append(tgt - trans)
            memory_list.append(memory)
        
        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        
        res_full = torch.cat(res, 2)
        input_res = torch.unsqueeze(res_full.permute(1, 0, 2), dim=1).transpose(2, 3)
        output = self.reslayer(input_res)
        recon2 = torch.squeeze(output).permute(0, 2, 1)
        return recon1[:, -recon2.shape[1]:, :], recon2


class ShortGraph(nn.Module):
    def __init__(self, dim, embed_time, slide_win, small_win):
        super(ShortGraph, self).__init__()
        self.name = 'ShortGraph'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = TransShort(embed_time, slide_win, small_win, 1)  # 这个1表示，每一个维度单独进入
        self.reslayer = Concurrent(dim, small_win, small_win, small_win)
    
    def forward(self, inputW, stage):
        if self.train:
            if stage == 1:
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = False
                for name, param in self.trans.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.trans.named_parameters():
                    param.requires_grad = False
                for name, param in self.reslayer.named_parameters():
                    param.requires_grad = True
        
        recon, res, origin, memory_list = [], [], [], []
        for i in range(1, inputW.shape[-1]):
            input_trans = inputW.permute(1, 0, 2)
            
            src_time = input_trans[:, :, 0].view(input_trans.shape[0], input_trans.shape[1], 1)
            src = input_trans[:, :, i].view(input_trans.shape[0], input_trans.shape[1], 1)
            
            trans = self.trans(src, src_time)
            recon.append(trans)
            origin.append(src)
            res.append(src - trans)
        
        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        origin_full = torch.cat(origin, 2).permute(1, 2, 0)
        res_full = torch.cat(res, 2).permute(1, 2, 0)
        recon2 = self.reslayer(res_full, origin_full)
        return recon1, recon2