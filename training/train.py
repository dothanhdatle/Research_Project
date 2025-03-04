import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..dataset.dataloader import SHREC22_data
from ..contrastive.moco_v3 import MoCo_v3, MoCo_model
from ..augmentations.augmentations import augmentations_sequence1, augmentations_sequence2
from ..encoders.encoders import STGCN_model

def train_(trainloader, model, augmentation, n_epochs, learning_rate, print_rate):
    train_loss = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(n_epochs)):
        # Training
        model.train()
        total_train_loss = 0.0
        for batch in trainloader:
            seqs = batch['Sequence'].float().to(model.device)
            seqs_aug1 = augmentation(seqs)
            seqs_aug2 = augmentation(seqs)
            optimizer.zero_grad()

            # Forward pass
            loss = model(seqs_aug1, seqs_aug2, training=True)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        train_loss.append(total_train_loss/len(trainloader))

        if (epoch + 1) % print_rate == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}]")
            print(f"  Train Loss: {train_loss[-1]:.4f}")
    
    return model, train_loss

