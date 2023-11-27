import torch
from torch import nn, Tensor
from torch.nn import functional as F


from .esg_utils import Dilated_Inception, MixProp, LayerNorm
from .graph import  NodeFeaExtractor, EvolvingGraphLearner


class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout:float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels,kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate  
        x = F.dropout(x, self.dropout, training=self.training)
        return x


# 重点，既包括了图结构学习EGL，又包括了MixProp
class Evolving_GConv(nn.Module):
    def __init__(self, conv_channels: int, residual_channels: int, gcn_depth: int,  st_embedding_dim: int, 
                dy_embedding_dim: int, dy_interval: int, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        
        # EGL
        self.scale_spc_EGL = EvolvingGraphLearner(conv_channels, dy_embedding_dim)
        self.dy_interval = dy_interval         

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)

    def forward(self, x, st_node_fea):
        # [4,16，321,181]
        b, _, n, t = x.shape
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)  #动态特征由静态特征直接转化来[321,40]--->[1,321,20]
        states_dy = dy_node_fea.repeat( b, 1, 1) #[B, N, C][4,321,20]

        x_out = []
    
        # range(0,181,31) 将输入序列的长度181，按照dy_interval进行切分
        for i_t in range(0,t,self.dy_interval):     
            x_i =x[...,i_t:min(i_t+self.dy_interval,t)]   # [4,16,321,31]
            # 聚合操作，达到r
            input_state_i = torch.mean(x_i.transpose(1,2),dim=-1)  # [4,321,16]
            # 将聚合后的r和states_dy输入到GRU，返回邻接矩阵和GRU新的状态states_dy
            dy_graph, states_dy= self.scale_spc_EGL(input_state_i, states_dy)
            x_out.append(self.gconv(x_i, dy_graph))    #[4,16,321,31]
        # 将每一段的输出x_out进行拼接，最后一个维度由dy_interval恢复为t
        x_out = torch.cat(x_out, dim= -1) #[B, c_out, N, T][4,16,321,181]
        return x_out


# 这是文中图2的三个模块之一
class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int, 
                st_embedding_dim, dy_embedding_dim, 
           skip_channels:int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))
      
        self.s_conv = Evolving_GConv(conv_channels, residual_channels, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                                    dy_interval, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
        
        self.norm = LayerNorm((residual_channels, num_nodes, t_len),elementwise_affine=layer_norm_affline)
       

    def forward(self, x: Tensor,  st_node_fea: Tensor):
        residual = x # [B, F, N, T][4,16,321,187]
        # dilated convolution
        x = self.t_conv(x)       
        # parametrized skip connection
        skip = self.skip_conv(x)
        # graph convolution[4,16,321,181]
        x = self.s_conv(x,  st_node_fea)         
        # residual connection
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        # return x
        return x, skip

# 这个就是paper中模型图中的三个层，或者说三个block之一
class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len : int, kernel_set, dilation_exp: int, n_layers: int, residual_channels: int, conv_channels: int,
    gcn_depth: int, st_embedding_dim, dy_embedding_dim,  skip_channels:int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1+ block_id*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            rf_block = block_id*n_layers*(kernel_size-1) + 1
        
        dilation_factor = 1
        for i in range(1, n_layers+1):            
            if dilation_exp>1:
                rf_size_i = int(rf_block + (kernel_size-1)*(dilation_exp**i-1)/(dilation_exp-1))
            else:
                rf_size_i = rf_block + i*(kernel_size-1)
            t_len_i = total_t_len - rf_size_i +1

            self.append(
                # residual_channels：16，conv_channels：16，kernel_set：7，dilation_factor：1，gcn_depth：2
                # skip_channels：32，t_len_i：181，layer_norm_affline：False,propalpha:0.05,dy_interval:[31,31,21,14,1]
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                 skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval[i-1])
            )
            dilation_factor *= dilation_exp


    def forward(self, x: Tensor, st_node_fea: Tensor, skip_list):
    # def forward(self, x: Tensor, st_node_fea: Tensor):
        flag = 0
        for layer in self:   # 遍历block里面的4个Extractor，也就是说这里的layer就是Extractor
            flag +=1
            # x是图2中GCN的输出，skip是GCN前面那个，DIL的输出
            x, skip = layer(x, st_node_fea)
            # x = layer(x, st_node_fea)
            # skip_list.append(skip)
        #     只返回最后一个GCN的输出和一个skip的list
        return x, skip_list
        # return x


class ESG(nn.Module):
    def __init__(self,                 
                 dy_embedding_dim: int,
                 dy_interval: list,
                 num_nodes: int,
                 seq_length: int,
                 pred_len : int,
                 in_dim: int,
                 out_dim: int,
                 n_blocks: int,
                 n_layers: int,                
                 conv_channels: int,
                 residual_channels: int,
                 skip_channels: int,
                 end_channels: int,
                 kernel_set: list,
                 dilation_exp: int,
                 gcn_depth: int,                                
                 # device,
                 fc_dim: int,
                 st_embedding_dim=40,
                 static_feat=None,
                 dropout=0.3,
                 propalpha=0.05,
                 layer_norm_affline=True
                 ):
        super(ESG, self).__init__()
       
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.num_nodes = num_nodes        
        # self.device = device
        self.pred_len = pred_len
        self.st_embedding_dim = st_embedding_dim
        self.seq_length = seq_length
        kernel_size = kernel_set[-1]
        if dilation_exp>1:
            self.receptive_field = int(1+n_blocks*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            self.receptive_field = n_blocks*n_layers*(kernel_size-1) + 1
            
        self.total_t_len = max(self.receptive_field, self.seq_length)
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        # 本文超参的block是1，Block内部分了5层
        for block_id in range(n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, kernel_set, dilation_exp, n_layers, residual_channels, conv_channels, gcn_depth,
                 st_embedding_dim, dy_embedding_dim, skip_channels, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval))

        self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.total_t_len), bias=True)
        # self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.total_t_len-self.receptive_field+1), bias=True)
        self.skipE = nn.Conv2d(residual_channels, 1, kernel_size=(1, 1), bias=True)
        # residual_channel:16   skip_channels:32
        in_channels = skip_channels
        final_channels = pred_len * out_dim

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, end_channels, kernel_size=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(end_channels, final_channels, kernel_size=(1,1), bias=True)     
        )
        self.stfea_encode = NodeFeaExtractor(st_embedding_dim, fc_dim)
        self.static_feat = static_feat
       

    def forward(self, input):
        """
        :param input: [B, in_dim, N, n_hist]
        :return: [B, n_pred, N, out_dim]
        """
        # [batch,1,dim,win] [128,1,24,60]
        b, _, n, t = input.shape
        assert t==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:    #[B,in_dim, N, receptive_field][4,1,321,187]
            input = F.pad(input,(self.receptive_field-self.seq_length,0,0,0), mode='replicate')
        
        x = self.start_conv(input)   #[B,16, N, receptive_field]
        # 其实就是对一个完整的训练数据进行卷积等操作降维[321, 40]
        st_node_fea = self.stfea_encode(self.static_feat)

        # [[4,32,321,1]]
        skip_list = [self.skip0(F.dropout(input, self.dropout, training=self.training))]
        for j in range(self.n_blocks):    
            x, skip_list= self.blocks[j](x, st_node_fea , skip_list)
            # x = self.blocks[j](x, st_node_fea)
        
        # xgraph [128,32,24,1]<---x[128,16,24,54]
        # # xgraph [128,1,24,54]<---x[128,16,24,54]
        # N：表示batch size（批处理参数） C_{in}：表示channel个数 H，W：分别表示特征图的高和宽
        xgraph = self.skipE(x)
        
        # skip_list.append(xgraph)
        # skip_list = torch.cat(skip_list, -1)                #[B, skip_channels, N, n_layers+2]
        # skip_sum = torch.sum(skip_list, dim=3, keepdim=True)  #[B, skip_channels, N, 1]
        # x = self.out(skip_sum) #[B, pred_len* out_dim, N, 1]
        # x = x.reshape(b, self.pred_len, -1, n).transpose(-1, -2) #[B, pred_len, N, out_dim]
        # return x
        return xgraph