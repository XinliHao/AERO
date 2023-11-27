# coding：utf-8
# @Time ：2022/11/8 16:43
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn

import torch
import torch.nn as nn
from models.temporal import Temporal
from models.concurrent import Concurrent


class AERO(nn.Module):
    def __init__(self, dim, embed_time,slide_win,small_win):
        super(AERO, self).__init__()
        self.name = 'AERO'
        self.slide_win = slide_win
        self.small_win = small_win
        self.trans = Temporal(embed_time,slide_win,small_win,1)   #这个1表示，每一个维度单独进入
        self.reslayer = Concurrent(dim, small_win, small_win, small_win)

    
    def forward(self, inputW,stage):
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
                
        recon,res,origin,memory_list = [],[],[],[]
        for i in range(1,inputW.shape[-1]):
            input_trans = inputW.permute(1,0,2)
            
            src_time = input_trans[:,:,0].view(input_trans.shape[0],input_trans.shape[1],1)
            src = input_trans[:,:,i].view(input_trans.shape[0],input_trans.shape[1],1)
            
            tgt = input_trans[-self.small_win:,:,i].view(self.small_win,input_trans.shape[1],1)
            
            trans, memory = self.trans(src, tgt, src_time)
            recon.append(trans)
            origin.append(tgt)
            res.append(tgt-trans)
            memory_list.append(memory)

        recon1 = torch.cat(recon, 2).permute(1, 0, 2)
        origin_full = torch.cat(origin, 2).permute(1, 2, 0)
        res_full = torch.cat(res, 2).permute(1, 2, 0)
        recon2 = self.reslayer(res_full, origin_full)
        return recon1, recon2