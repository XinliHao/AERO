# coding：utf-8
# @Time ：2022/10/26 10:28
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn

from src.dataloader import split_dataset
from src.dataloader import convert_to_windows
from src.dataloader import StandardScaler
from train import train
from test import test
import torch.nn as nn
import models.AERO
import ablation.Ablation

from pprint import pprint
from src.plot import *
import pandas as pd
from datetime import datetime
from src.evaluate import *
from parser import *
from torch.utils.data import DataLoader
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')



class Main():
    def __init__(self):
        # ## Prepare data
        self.train, self.val, self.test, self.labels = split_dataset(env_config['dataset_name'],env_config['data_folder'],env_config['model_name'])
        
        self.trainW = convert_to_windows(self.train,model_config['slide_win'])
        self.valiW =convert_to_windows(self.val,model_config['slide_win'])
        self.testW = convert_to_windows(self.test,model_config['slide_win'])
        
        self.trainW_loader = DataLoader(self.trainW, batch_size=train_config['batch_size'])
        self.valiW_loader = DataLoader(self.valiW, batch_size=train_config['batch_size'])
        self.testW_loader = DataLoader(self.testW, batch_size=train_config['batch_size'])
    
        self.dims = self.labels.shape[1]
        self.build_model()
        
    def build_model(self):
        # load model or build new model
        if env_config['model_name'] == 'AERO':
            model_class = getattr(models.AERO, env_config['model_name'])
            self.model = model_class(self.dims, model_config['embed_time'], model_config['slide_win'], model_config['small_win']).to(env_config['device'])
        
        elif env_config['model_name'] == 'OnlyTemporal' or env_config['model_name'] == 'OnlyTemporalMulti' or env_config['model_name'] == 'StaticGraph' or env_config['model_name'] == 'MultiVariate':
            model_class = getattr(ablation.ablation, env_config['model_name'])
            self.model = model_class(self.dims, model_config['embed_time'], model_config['slide_win'], model_config['small_win']).to(env_config['device'])
        
        elif env_config['model_name'] == 'OnlyConcurrent':
            model_class = getattr(ablation.ablation, env_config['model_name'])
            self.model = model_class(self.dims, model_config['slide_win'], model_config['small_win']).to(env_config['device'])
        
        if env_config['model_name'] == 'ShortGraph':
            model_class = getattr(ablation.ShortGraph, env_config['model_name'])
            self.model = model_class(self.dims, model_config['embed_time'], model_config['slide_win'], model_config['small_win']).to(env_config['device'])
        
        elif env_config['model_name'] == 'DynamicGraph':
            data_set = env_config['dataset_name']
            folder = os.path.join('./processed/', data_set)
            finalpath = os.path.join(folder, f'{data_set}_train.npy')
            train_data = np.load(finalpath)[:, 1:]
            scaler = StandardScaler(np.mean(train_data, axis=0), np.std(train_data, axis=0))
            train_feas = scaler.transform(train_data)
            train_feas = torch.tensor(train_feas).type(torch.FloatTensor).to(env_config['device'])
            model_class = getattr(ablation.ablation, env_config['model_name'])
            self.model = model_class(model_config['embed_time'], model_config['slide_win'], model_config['small_win'],model_config['fc_dim'],train_feas).to(env_config['device'])
        
        for name, param in self.model.named_parameters():
            if 'reslayer.mask' in name:
                nn.init.ones_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

        update_parms = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(update_parms, lr=train_config['lr'], amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5,threshold=1e-4)
        
        fname = 'checkpoints/{}_{}/model.ckpt'.format(env_config['model_name'], env_config['dataset_name'])
        if os.path.exists(fname) and (not train_config['retrain'] or train_config['test']):
            print(f"Loading pre-trained model: {self.model.name}")
            checkpoint = torch.load(fname)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.accuracy_list = checkpoint['accuracy_list']
        else:
            print(f"Creating new model: {self.model.name}")
            self.epoch = -1;
            self.accuracy_list = []
    
    def run(self):
        # 加载模型
        self.build_model()
        if train_config['retrain']:
            train(env_config,train_config,model_config,self.model,self.optimizer,self.scheduler,self.epoch,self.accuracy_list,self.trainW_loader,self.valiW_loader)
        
        fname = 'checkpoints/{}_{}/model.ckpt'.format(env_config['model_name'], env_config['dataset_name'])
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        best_model = self.model.to(env_config['device'])
        
        resultT = test(best_model, self.trainW_loader, env_config, model_config)
        result = test(best_model, self.testW_loader, env_config,model_config)
        
        self.plot_save(result,self.labels,resultT)

    def plot_save(self,result,labels,resultT):
        
        result_flat, predict_label_flat, f1_list_flat, f1auc_flat = flat_pot_eval(result['loss12'], result['loss12'], labels[model_config['slide_win']:], q=model_config['q'],level=model_config['level'])
        print('=' * 30 + 'result' + '=' * 30)
        pprint(result_flat)
    
        if not os.path.exists(os.path.join(env_config['output_folder'],'result.csv')):
            df = pd.DataFrame(columns=['time', 'dataset_name', 'model_name','f1','precision','recall','TP','TN','FP','FN','threshold',
                                       'test','retrain','batch_size','epoch_num','init_lr','freeze_patience','freeze_delta','stop_patience','stop_delta',
                                       'slide_win','small_win','embed_time','level','q'])
            df.to_csv(os.path.join(env_config['output_folder'],'result.csv'), index=False)
        
        save_result = [datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),env_config['dataset_name'],env_config['model_name'],
                       '%.4f'%result_flat['f1'],'%.4f'%result_flat['precision'],'%.4f'%result_flat['recall'],
                       '%d'%result_flat['TP'],'%d'%result_flat['TN'],'%d'%result_flat['FP'],'%d'%result_flat['FN'],'%.4f'%result_flat['threshold'],
                       train_config['test'],train_config['retrain'],train_config['batch_size'],train_config['epoch_num'],train_config['lr'],
                       train_config['freeze_patience'],train_config['freeze_delta'],train_config['stop_patience'],train_config['stop_delta'],
                       model_config['slide_win'],model_config['small_win'],model_config['embed_time'],model_config['level'],model_config['q']]
        data = pd.DataFrame([save_result])
        data.to_csv(os.path.join(env_config['output_folder'],'result.csv'),mode='a',header=False,index=False)
        
        
if __name__ == '__main__':
    
    # 消除随机性
    seed = 20222022
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    env_config,train_config,model_config = myconfig()
    
    main = Main()
    main.run()