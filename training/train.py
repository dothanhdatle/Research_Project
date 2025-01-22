import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..dataset.dataloader import SHREC22_data
from ..contrastive.moco import MoCo
from ..augmentations.augmentations import augmentations_sequence1, augmentations_sequence2
from ..encoders.encoders import STGCN_model


# Hyperparameters
train_path = './dataset/training_set/training_set/'
test_path = './dataset/training_set/test_set/'
T = 90  # Sequence length4

seed=42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train data
trainset = SHREC22_data(train_path, T)
# Extract labels and indices
labels = [data['Label'] for data in trainset]
indices = list(range(len(trainset)))

# train validation split
train_indices, val_indices = train_test_split(
    indices, 
    test_size=0.2,    
    stratify=labels,  
    random_state=seed   
)

# Create subsets for training and validation
train_set = Subset(trainset, train_indices)
val_set = Subset(trainset, val_indices)

print('Train set size:',len(train_set))
print('Validation set size:',len(val_set))

# Hyperparameters for self-supervised training
batch_size = 64
learning_rate = 1e-3
queue_size = 10000
graph_args = {
    'strategy': 'spatial',
    'max_hop': 1,
    'dilation': 1
}

momentum = 0.999 # momentum
temperature = 0.07 # temperature

# Data loader for self-supervised without labels
trainloader_ssl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader_ssl = DataLoader(val_set, batch_size=batch_size, shuffle=True)
# Data augmentation
augmentations1 = augmentations_sequence1()
augmentations2 = augmentations_sequence2()

# Load base encoder for MoCo
base_encoder = STGCN_model(in_channels=3, 
                           hidden_channels=16, 
                           hidden_dim=64, out_dim=128, 
                           graph_args=graph_args, 
                           edge_importance_weighting=True,
                           dropout_rate=0).to(device)

# Load MoCo Model
moco_model = MoCo(base_encoder=base_encoder, 
                  dim = 128, 
                  K=queue_size, 
                  m=momentum, 
                  T=temperature,
                  mlp=False).to(device)

n_epochs = 1000
optimizer = optim.Adam(moco_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
# Training Loop
for epoch in tqdm(range(n_epochs)):
    moco_model.train()
    total_loss = 0.0
    for batch in trainloader_ssl:
        seqs = batch['Sequence'].float().to(device)
        seqs_aug1 = augmentations1(seqs)
        seqs_aug2 = augmentations1(seqs)

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

