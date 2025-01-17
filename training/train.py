import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset.dataloader import SHREC22_data
from ..contrastive.moco import MoCo
from ..augmentations.augmentations import augmentations_sequence
from ..encoders.encoders import STGCN_model


# Hyperparameters
train_path = './dataset/training_set/training_set/'
batch_size = 64
n_epochs = 100
learning_rate = 0.001
queue_size = 8192
momentum = 0.999
temperature = 0.07
T = 90  # Sequence length4
graph_args = {
    'strategy': 'spatial',
    'max_hop': 1,
    'dilation': 1
}

seed=42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train data
trainset = SHREC22_data(train_path, T=T)
train_ssl_size = int(0.9 * len(trainset))
train_classify_size = len(trainset) - train_ssl_size
train_ssl_set, train_classify_set = torch.utils.data.random_split(trainset, [train_ssl_size, train_classify_size], generator=torch.Generator().manual_seed(seed))
trainloader_ssl = DataLoader(train_ssl_set, batch_size=batch_size, shuffle=True)


# Data augmentation
augmentations = augmentations_sequence()

# Load base encoder for MoCo
base_encoder = STGCN_model(in_channels=3, hidden_channels=64, hidden_dim=256, out_dim=256, graph_args=graph_args, edge_importance_weighting=True)

# Load MoCo Model
moco_model = MoCo(base_encoder=base_encoder, dim = 256, K=queue_size, m=momentum, T = temperature).to(device)

# Optimizer
optimizer = optim.Adam(moco_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

moco_model.train()
# Training Loop
for epoch in tqdm(range(n_epochs)):
    total_loss = 0.0
    for batch in trainloader_ssl:
        seqs = batch['Sequence'].float().to(device)
        seqs_aug1 = augmentations(seqs)
        seqs_aug2 = augmentations(seqs)

        # Forward pass
        logits, labels = moco_model(seqs_aug1, seqs_aug2)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(trainloader_ssl):.4f}")

# Save the model
torch.save(moco_model.state_dict(), 'moco_stgcn.pth')

