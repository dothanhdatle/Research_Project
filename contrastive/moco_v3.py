import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MoCo_v3(nn.Module):
    """ Referring to the paper of MoCo v3: https://arxiv.org/abs/2104.02057v4 """
    def __init__(self, 
                 base_encoder, 
                 dim=128, 
                 mlp_dim=256, 
                 m=0.999, 
                 T=0.07, 
                 T_max = 0.2, 
                 T_min = 0.05, 
                 T_type='fix'):
        """
        Params:
            dim: feature dimension (default: 128)
            mlp_dim: hidden dimension in MLPs (default: 256)
            m: momentum (default: 0.999)
            T: temperature (default: 0.07)
            T_max: max temperature for dynamic temperature (default: 0.2)
            T_min: min temperature for dynamic temperature (default: 0.05)
            T_type: fixed temperature or dynamic temperature refers to paper: https://arxiv.org/pdf/2308.01140v2
        """
        super(MoCo_v3, self).__init__()

        self.m = m
        self.T_type = T_type

        if self.T_type == 'fix':
            self.T = T
        else:
            self.T_min = T_min
            self.T_max = T_max

        # build encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        self.build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # Initialize parameters of the 2 encoders
            param_k.data.copy_(param_q.data)
            # Not update the keys encoder by gradient
            param_k.requires_grad = False
    
    def build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    def contrastive_loss(self, q, k):
        """ Contrastive loss """
        # Normalize embeddings
        q = F.normalize(q, dim=1)  
        k = F.normalize(k, dim=1)  
        
        # Compute cosine similarity matrix
        cosine_sim = torch.einsum('nc,mc->nm', [q, k])  # Shape: (N, N)
        
        # Compute temperatures 
        if self.T_type == 'fix':
            tau = self.T
        else:
            tau = self.T_min + 0.5 * (self.T_max - self.T_min) * (1 + torch.cos(torch.pi * (1 + cosine_sim)))  # Shape: (N, N)
        
        # Compute logits 
        logits = cosine_sim / tau

        # Compute contrastive loss
        N = logits.shape[0]  # batch size
        labels = torch.arange(N, dtype=torch.long, device=q.device)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        if self.T_type == 'fix':
            return loss * (2 * tau)
        else:
            return loss * (2 * tau.mean())
    
    def forward(self, x1, x2, training=True):
        """
        Input:
            x1: first augmentation
            x2: second augmentation
        Output:
            loss
        """

        # compute queries
        q1 = self.predictor(self.encoder_q(x1))
        q2 = self.predictor(self.encoder_q(x2))

        with torch.no_grad():  # no gradient
            if training:
                self._momentum_update_key_encoder()  # update the momentum encoder

            # compute keys
            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)
        
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1) # symmetrized loss

        return loss
    
class MoCo_model(MoCo_v3):
    def build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_q.fc.weight.shape[1]
        del self.encoder_q.fc, self.encoder_k.fc # remove original fc layer in the encoder

        # projectors
        self.encoder_q.fc = self.build_mlp(2, hidden_dim, mlp_dim, dim)
        self.encoder_k.fc = self.build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self.build_mlp(2, dim, mlp_dim, dim, False)
    




