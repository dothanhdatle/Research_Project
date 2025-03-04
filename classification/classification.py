import torch
import torch.nn as nn
import torch.nn.functional as F

# SSL framework with a simple classifier layer for downstream task 
class SSL_model(nn.Module):
    def __init__(self, contrastive_encoder, classifier_dim, num_classes = 16):
        super(SSL_model, self).__init__()
        self.contrastive_encoder = contrastive_encoder
        self.classifier_dim = classifier_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.classifier_dim, self.num_classes)

        for param in self.contrastive_encoder.parameters():
            param.requires_grad = False
        
    def forward(self,x):
        self.contrastive_encoder.eval()
        with torch.no_grad():
            feature = self.contrastive_encoder(x)
        
        output = self.classifier(feature)
        return output
    
class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
    
    def forward(self,x):
        logits = self.classifier(x)
        return logits
    


