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

def test_(testloader, contrastive_model, classifier_model):
    for param in contrastive_model.encoder_q.parameters():
        param.requires_grad = False

    contrastive_model.encoder_q.eval()
    classifier_model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in testloader:
            seqs = batch['Sequence'].float().to(contrastive_model.device)
            labels = batch['Label'].to(contrastive_model.device)

            # Forward pass
            feature = contrastive_model.encoder_q(seqs)
            output = classifier_model(feature)

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy