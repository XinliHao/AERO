# -*-coding:utf-8 -*-

# @Time ：2023/6/13 15:37
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn

import torch
import torch.nn as nn
from parser import color


def valid(model, data_loader, env_config, model_config):
    print('{}Validating {} on {}{}'.format(color.HEADER, env_config['model_name'], env_config['dataset_name'], color.ENDC))
    if 'AERO' in env_config['model_name'] or 'StaticGraph' in env_config['model_name'] \
        or env_config['model_name'] == 'MultiVariate' or env_config['model_name'] == 'DynamicGraph' or env_config['model_name'] == 'ShortGraph':
        lossFunction = nn.MSELoss(reduction='none')
        torch.zero_grad = True
        model.eval()
        loss12_list, loss1_list = [], []
        stage = 2
        for d in data_loader:
            with torch.no_grad():
                recon1, recon2 = model(d, stage)
            short_data = d[:, -recon2.shape[1]:, 1:]
            loss12 = torch.mean(lossFunction(recon1 + recon2, short_data)).cpu().detach().numpy()
            loss12_list.append(loss12)
        return loss12_list

    if env_config['model_name'] == 'OnlyTemporalMulti' or env_config['model_name']=='OnlyTemporal'  or env_config['model_name']=='OnlyConcurrent':
        lossFunction = nn.MSELoss(reduction='none')
        torch.zero_grad = True
        model.eval()
        loss1_list = []
        for d in data_loader:
            with torch.no_grad():
                recon1 = model(d)
            short_data = d[:, -model_config['small_win']:, 1:]
            loss1 = torch.mean(lossFunction(recon1, short_data)).cpu().detach().numpy()
            loss1_list.append(loss1)
        return loss1_list