# coding：utf-8
# @Time ：2022/10/26 10:28
# @Author:xinli hao
# @Email:xinli_hao@ruc.edu.cn

import argparse
import torch

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def myconfig():
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    
    # env.config
    parser.add_argument('--dataset_name',metavar='-d',type=str,required=False,default='SyntheticMiddle',help="dataset name")
    parser.add_argument('--model_name', metavar='-m', type=str, required=False, default='AERO')

    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--output_folder', metavar='-o', type=str, required=False, default='./output')
    parser.add_argument('--data_folder', metavar='-o', type=str, required=False, default='./processed')
    
    # train config
    parser.add_argument('--test', action='store_true', help="test the model")
    parser.add_argument('--retrain', action='store_true', help="retrain the model")
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--epoch_num',type=int,default=100)
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze_patience',type=int,default=5)
    parser.add_argument('--freeze_delta',type=float,default=0.01)
    
    parser.add_argument('--stop_patience',type=int,default=5)
    parser.add_argument('--stop_delta',type=float,default=0.005)

    # model config
    parser.add_argument('--slide_win',type=int,default=200)
    parser.add_argument('--small_win',type=int,default=60)
    parser.add_argument('--embed_time', type=int, default=8)
    
    parser.add_argument('--level', type=float, default=0.99)
    parser.add_argument('--q', type=float, default=0.001)
    
    parser.add_argument('--bf_search_min', default=0, type=float)
    parser.add_argument('--bf_search_max', default=0.5, type=float)
    parser.add_argument('--bf_search_step_size', default=0.001, type=float)
    parser.add_argument('--fc_dim', type=int, default=0)
    
    args = parser.parse_args()
    
    fc_dim_dir = {
            'SyntheticMiddle': 63712,
            'SyntheticHigh': 63712,
            'SyntheticLow':63712,
            'AstrosetMiddle':88352,
            'AstrosetHigh': 112832,
            'AstrosetLow':99792,
    }
    
    args.fc_dim = fc_dim_dir[args.dataset_name]
    
    env_config = {
        'dataset_name': args.dataset_name,
        'model_name':args.model_name,
        'device': args.device,
        'output_folder':args.output_folder,
        'data_folder': args.data_folder,
    }
    
    train_config = {
        'test':args.test,
        'retrain':args.retrain,
        'batch_size': args.batch_size,
        'epoch_num': args.epoch_num,
        'lr':args.lr,
        'freeze_patience':args.freeze_patience,
        'freeze_delta':args.freeze_delta,
        'stop_patience': args.stop_patience,
        'stop_delta': args.stop_delta,
    }
    
    model_config = {
        'slide_win':args.slide_win,
        'small_win':args.small_win,
        'embed_time':args.embed_time,
        'level':args.level,
        'bf_search_min':args.bf_search_min,
        'bf_search_max':args.bf_search_max,
        'bf_search_step_size':args.bf_search_step_size,
        'fc_dim':args.fc_dim
    }
    
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    return env_config,train_config,model_config