import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MoCo(nn.Module):
    """ Referring to the paper of MoCo: https://arxiv.org/abs/1911.05722 """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        base_encoder: the encoder network to encode keys and queries
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoder for keys and queries
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # Initialize parameters of the 2 encoders
            param_k.data.copy_(param_q.data)
            # Not pdate the keys encoder by gradient
            param_k.requires_grad = False

        # Create the queue size K to contain negative keys
        self.register_buffer("queue", torch.randn(dim,K)) # size K
        self.queue = F.normalize(self.queue, dim=0)

        # Pointer indicating the position in queue to dequeue and enqueue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue and enqueue the negative keys
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        """ assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer """

        ## Without assertion
        # Ensure enough space to enqueue
        if ptr + batch_size <= self.K:
            # If there is enough space in the queue, enqueue directly
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        else:
            # If not enough space in queue for a batch size
            # Fill the remaining space first
            remaining_space = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining_space].T

            # Wrap around and fill the start of the queue
            self.queue[:, :batch_size - remaining_space] = keys[remaining_space:].T
            ptr = batch_size - remaining_space

        # Update the pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, training = True):
        """
        Input:
            im_q: a batch of query
            im_k: a batch of key 
        """

        # compute query features by the query encoder
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        #print("Query grad_fn:", q.grad_fn)  e

        # compute key features by the key encoder
        with torch.no_grad():  # no gradient to keys
            if training:
                self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            #print("Key grad_fn (should be None):", k.grad_fn)  

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        #print("l_pos grad_fn:", l_pos.grad_fn)  
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        #print("l_neg grad_fn:", l_neg.grad_fn)  

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        #print("Logits grad_fn:", logits.grad_fn)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        #print("Labels grad_fn:", labels.grad_fn)  # Should be None, as labels are constant

        # dequeue and enqueue
        if training:
            self._dequeue_and_enqueue(k)

        return logits, labels




