# -*-coding:utf-8 -*-

# @Time ：2023/6/11 16:45
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn
import numpy as np
import torch
import os
from parser import color
from time import time
from tqdm import tqdm
import torch.nn as nn
from valid import valid

class EarlyStopping:
    def __init__(self, patience=2, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_freeze = False
        self.delta = delta

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss < self.delta:
        # 这个在最后的消融实验之前是没有大于零这个约束的，只在032_22790425-G0014这个数据集上
        # elif self.best_loss - loss < self.delta  and self.best_loss - loss > 0 :
            self.counter += 1
            print("diff",f'{self.best_loss - loss}')
            print(f'Early counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_freeze = True
        else:
            self.best_loss = loss
            self.counter = 0

def save_model(model, optimizer, scheduler, epoch, accuracy_list,folder):
    print('{}Saving model ...{}'.format(color.HEADER,color.ENDC))
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def train(env_config,train_config,model_config,model,optimizer,scheduler,pre_epoch,accuracy_list,data_loader,vali_loader):
    print('{}Training {} on {}{}'.format(color.HEADER, env_config['model_name'], env_config['dataset_name'],color.ENDC))
    num_epochs = train_config['epoch_num'];
    start = time()
    stage = 1
    early_freezing = EarlyStopping(patience=train_config['freeze_patience'], delta=train_config['freeze_delta'])
    early_stopping = EarlyStopping(patience=train_config['stop_patience'], delta=train_config['stop_delta'])
    
    if env_config['model_name'] == 'AERO' or env_config['model_name'] == 'StaticGraph' \
        or env_config['model_name'] == 'MultiVariate' or env_config['model_name'] == 'DynamicGraph' or env_config['model_name'] == 'ShortGraph':
        for e in range(pre_epoch + 1, pre_epoch + num_epochs + 1):
            lossFunction = nn.MSELoss(reduction='none')
            losssave = []
    
            print("\n===========================")
            print("第%d个epoch开始，当前学习率：%f, 当前stage:%d" % (e, optimizer.param_groups[0]['lr'], stage))
            for i, d in enumerate(data_loader):  # [B,slide_win,feat]
                cuda_d = d.float().to(env_config['device'])
                recon1, recon2 = model(cuda_d,stage)
                short_data = cuda_d[:, -recon2.shape[1]:, 1:]
                if stage == 1:
                    a = lossFunction(recon1, short_data)
                    loss_batch_mean = torch.mean(a)
                    losssave.append(loss_batch_mean.item())
                    loss_batch_mean.backward()
                else:
                    last_time_loss = lossFunction(recon1 + recon2, short_data)
                    loss_batch_mean = torch.mean(last_time_loss)
                    losssave.append(loss_batch_mean.item())
                    loss_batch_mean.backward()
        
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step(loss_batch_mean)   #更新lr
            loss_epoch_mean = np.mean(losssave)
            accuracy_list.append((loss_epoch_mean, optimizer.param_groups[0]['lr']))
            tqdm.write(f'Epoch {e},\tL1 = {loss_epoch_mean}')

            if stage == 1:
                early_freezing(loss_epoch_mean)
                if early_freezing.early_freeze:
                    print("Early freezing")
                    stage = 2
            
            if stage ==2:
                if vali_loader is not None:
                    avg_vali_loss = np.mean(valid(model, vali_loader, env_config, model_config))
                    print('avg_vali_loss',avg_vali_loss)
                    early_stopping(avg_vali_loss)
                    if early_stopping.early_freeze:
                        print("Early stopping")
                        break
                else:
                    early_stopping(loss_epoch_mean)
                    if early_stopping.early_freeze:
                        print("Early stopping")
                        break
    
    elif env_config['model_name'] == 'OnlyTemporalMulti' or env_config['model_name'] == 'OnlyTemporal'or env_config['model_name'] == 'OnlyConcurrent':
        for e in range(pre_epoch + 1, pre_epoch + num_epochs + 1):
            lossFunction = nn.MSELoss(reduction='none')
            losssave = []
            print("\n===========================")
            print("第%d个epoch开始，当前学习率：%f" % (e, optimizer.param_groups[0]['lr']))
            for i, d in enumerate(data_loader):
                cuda_d = d.float().to(env_config['device'])
                recon1 = model(cuda_d)
                short_data = cuda_d[:, -model_config['small_win']:, 1:]
                loss_batch_mean = torch.mean(lossFunction(recon1, short_data))
                losssave.append(loss_batch_mean.item())
                loss_batch_mean.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step(loss_batch_mean)  # 更新lr
            loss_epoch_mean = np.mean(losssave)
            accuracy_list.append((loss_epoch_mean, optimizer.param_groups[0]['lr']))
            tqdm.write(f'Epoch {e},\tL1 = {loss_epoch_mean}')
            if vali_loader is not None:
                avg_vali_loss = np.mean(valid(model, vali_loader, env_config, model_config))
                print('avg_vali_loss', avg_vali_loss)
                early_stopping(avg_vali_loss)
                if early_stopping.early_freeze:
                    print("Early stopping")
                    break
            else:
                early_stopping(loss_epoch_mean)
                if early_stopping.early_freeze:
                    print("Early stopping")
                    break

    print('Training and validing time: ' + "{:10.4f}".format(time() - start) + ' s')
    folder = 'checkpoints/{}_{}'.format(env_config['model_name'], env_config['dataset_name'])
    save_model(model, optimizer, scheduler, e, accuracy_list, folder)

    # 学习率、损失变化图
    # plot_accuracies(accuracy_list, '{}_{}_{}'.format(env_config['model_name'], env_config['dataset_name'],env_config['graph_type']))
    
    