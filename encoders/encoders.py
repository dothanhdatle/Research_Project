import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from encoders.utils.tgcn import ConvTemporalGraphical
from encoders.utils.graph import Graph


## ST-GCN as encoder
class STGCN_model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        hidden_channels (int): Number of hidden channels
        hidden_dim (int): Number of hidden dimensions
        out_dim (int): Number of output dimension
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, out_dim)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes
    """

    def __init__(self, in_channels, hidden_channels, hidden_dim, out_dim , graph_args,
                 edge_importance_weighting):
        super(STGCN_model, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False) # adjacency matrices
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)  # The size of the spatial kernel (Number of adjacency matrices)
        temporal_kernel_size = 9  # The size of the temporal kernel
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # Batch normalization 
        self.st_gcn_net = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, stride=1, residual=False),  # (N, hidden_channels, T, V)
            st_gcn(hidden_channels, hidden_channels, kernel_size, stride=1),
            #st_gcn(hidden_channels, hidden_channels, kernel_size, stride=1),
            #st_gcn(hidden_channels, hidden_channels, kernel_size, stride=1),
            st_gcn(hidden_channels, hidden_channels*2, kernel_size, stride=2),  # (N, hidden_channels, T/2, V)
            st_gcn(hidden_channels*2, hidden_channels*2, kernel_size, stride=1),
            #st_gcn(hidden_channels*2, hidden_channels*2, kernel_size, stride=1),
            st_gcn(hidden_channels*2, hidden_channels*4, kernel_size, stride=2),  # (N, hidden_channels, T/4, V)
            #st_gcn(hidden_channels*4, hidden_channels*4, kernel_size, stride=1),
            st_gcn(hidden_channels*4, hidden_dim, kernel_size, stride=1), # (N, hidden_dim, T/4, V)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))  # learnable importance weights of spatial edges
                for i in self.st_gcn_net
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_net)

        #self.fc = nn.Linear(hidden_dim, out_dim) 
        self.fc = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x):

        # Input data (N,T,V,C)
        N, T, V, C= x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,V,C,T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x) 
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,C,T,V)

        # forwad
        for gcn, importance in zip(self.st_gcn_net, self.edge_importance):
            x, _ = gcn(x, self.A * importance)  # (N, hidden_dim, T/4, V)

        # global pooling both the temporal and spatial dimensions
        x = F.avg_pool2d(x, x.size()[2:]) # (N, hidden_dim, 1, 1)
        #x = x.view(N, -1)

        # prediction
        x = self.fc(x)  # (N, out_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (N, out_dim)

        return x

    """ def extract_feature(self, x):

        # Input data (N,T,V,C)
        N, T, V, C= x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,V,C,T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x) 
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,C,T,V)

        # forwaRd
        for gcn, importance in zip(self.st_gcn_net, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        feature = x # (N, hidden_dim, T/4, V)

        output = self.fc(x)# (N, out_dim, t, v)

        return output, feature """

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and spatial convolving kernel (temporal size, spatial size)
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual connection mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2   # (temporal kernel size, spatial kernel size)
        assert kernel_size[0] % 2 == 1  # temporal kernel size is odd
        padding = ((kernel_size[0] - 1) // 2, 0)  # padding for temporal convolution

        # Spatial Graph Convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        
        # Temporal Convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
    

## Transformers encoder
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 

        self.register_buffer("pe", pe)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x):
        return x + self.position_embedding[:, :x.size(1)]
    
class TransformersEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, max_len=5000, norm_first=True, dropout=0.1):

        super(TransformersEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        #self.positional_encoding = LearnablePositionalEncoding(embed_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                    nhead=num_heads, 
                                                    dim_feedforward=embed_dim*4,
                                                    dropout=dropout, 
                                                    batch_first=True,
                                                    norm_first=norm_first)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Attention-based Temporal Pooling

    def forward(self, x):
        
        x = self.embedding(x)
        x = self.positional_encoding(x)

        out = self.transformer_encoder(x)  # (N,T,embed_dim)
        out = out.permute(0,2,1)  # (N,embed_dim,T)
        out = self.pool(out).squeeze(-1) # (N,embed_dim) 

        return out

## LSTM encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.fc(out[:, -1, :])  # Take last time step output
        return out

## GRU encoder
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)  
        out = self.fc(out[:, -1, :])  # Take last time step output
        return out  


    
