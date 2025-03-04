import random

import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2


class Rotate(nn.Module):
    def __init__(self, axis='z', angle=None):
        super().__init__()
        self.axis = axis
        self.angle = angle

    def forward(self, sequences):
        N, T, V, C = sequences.shape
        device = sequences.device
        temp = sequences.clone()

        # Convert angle to radians
        angle_rad = torch.deg2rad(torch.tensor(self.angle, dtype=sequences.dtype, device=device))
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

        # Define rotation matrices for the specified axis
        if self.axis == 'x':  # Rotate around X-axis
            R = torch.tensor([[1, 0, 0],
                              [0, cos_a, -sin_a],
                              [0, sin_a, cos_a]], device=device, dtype=sequences.dtype)

        elif self.axis == 'y':  # Rotate around Y-axis
            R = torch.tensor([[cos_a, 0, sin_a],
                              [0, 1, 0],
                              [-sin_a, 0, cos_a]], device=device, dtype=sequences.dtype)

        elif self.axis == 'z':  # Rotate around Z-axis
            R = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0],
                              [0, 0, 1]], device=device, dtype=sequences.dtype)

        # Apply the rotation using matrix multiplication across the batch
        temp = temp.view(-1, C)
        temp = torch.matmul(temp, R.T)
        temp = temp.view(N, T, V, C)

        return temp

class Shear(nn.Module):
    def __init__(self, s1=None, s2=None):
        super().__init__()
        self.s1 = s1
        self.s2 = s2

    def forward(self, sequences):
        # Create a clone of the tensor to avoid modifying the original data
        temp = sequences.clone()
        device = sequences.device

        # Generate random shear values if not provided
        if self.s1 is not None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        
        if self.s2 is not None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        # Construct the shear matrix
        shear_matrix = torch.tensor([
            [1, s1_list[0], s2_list[0]],  
            [s1_list[1], 1, s2_list[1]],  
            [s1_list[2], s2_list[2], 1]   
        ], device=device, dtype=sequences.dtype) 

        # Apply shear transformation
        temp = torch.einsum('ntvc,cd->ntvd', temp, shear_matrix.T)

        return temp
    
class RandomHorizontalFlip(nn.Module):
    def __init__(self):
        super().__init__()
        self.ok = True

    def forward(self, sequences):
        temp = sequences.clone()
        # Flip along the X-axis
        temp[..., 0] *= -1 

        return temp

class JointSubtract(nn.Module):
    def __init__(self, joint=1):
        super().__init__()
        # Default joint is the second joint (index 1)
        self.joint = joint

    def forward(self, sequences):
        # data_tensor shape: (N, T, V, C)
        N, T, V, C = sequences.shape
        
        # Subtract the data of the selected joint (self.joint) from all other joints
        # We use broadcasting to subtract across all the joints for each batch, time step, and channel
        reference_joint = sequences[:, :, self.joint, :].unsqueeze(2)  # Shape: (N, T, 1, C)
        new_seqs = sequences - reference_joint  # Broadcasting subtraction: (N, T, V, C)
        
        return new_seqs

class RandomPerspectiveBatch(nn.Module):
    def __init__(self, max_angle=30):
        super().__init__()
        self.max_angle= max_angle

    def forward(self, sequences):
        device = sequences.device
        angles = torch.tensor([
            random.uniform(-self.max_angle, self.max_angle) for _ in range(3)
        ])
        angles_rad = torch.deg2rad(angles)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles_rad[0]), -torch.sin(angles_rad[0])],
            [0, torch.sin(angles_rad[0]), torch.cos(angles_rad[0])]
        ], device = device, dtype=sequences.dtype)
        Ry = torch.tensor([
            [torch.cos(angles_rad[1]), 0, torch.sin(angles_rad[1])],
            [0, 1, 0],
            [-torch.sin(angles_rad[1]), 0, torch.cos(angles_rad[1])]
        ], device = device, dtype=sequences.dtype)
        Rz = torch.tensor([
            [torch.cos(angles_rad[2]), -torch.sin(angles_rad[2]), 0],
            [torch.sin(angles_rad[2]), torch.cos(angles_rad[2]), 0],
            [0, 0, 1]
        ], device = device, dtype=sequences.dtype)
        

        R = Rz @ Ry @ Rx
        
        N, T, V, C = sequences.shape
        sequences = sequences.view(-1, C) @ R.T
        sequences = sequences.view(N, T, V, C)
        
        return sequences

class RandomTemporalCropAndPad(nn.Module):
    def __init__(self, min_crop_length):
        super().__init__()
        self.min_crop_length = min_crop_length

    def forward(self, sequences):
        N, T, V, C = sequences.shape

        # Determine a random crop length between min_crop_length and T
        crop_length = random.randint(self.min_crop_length, T-1)

        # Crop from the first frame
        cropped = sequences[:, :crop_length, :, :]

        # Zero-pad to recover the original length
        new_seqs = F.pad(cropped, (0, 0, 0, 0, 0, T - crop_length), mode="constant", value=0)

        return new_seqs

class HideSequenceSegment(nn.Module):
    def __init__(self, min_ratio=0.1, max_ratio=0.3):
        super().__init__()

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio


    def forward(self, sequences):
        
        n_frames = 90
        batch_size = sequences.shape[0]
        for i in range(batch_size):
            hide_len = int(n_frames * random.uniform(self.min_ratio, self.max_ratio))
            start = random.randint(0, n_frames - hide_len)
            sequences[i, start:start + hide_len] = 0
        
        return sequences

class VideoAccelerate(nn.Module):
    def __init__(self):
        super().__init__()
        self.ok = True


    def forward(self, sequences):

        N, T, V, C = sequences.shape
        
        indices = torch.arange(0, T, 2)
        sequences = sequences[:, indices, :, :]
        
        n_missing = T - sequences.shape[1]
        if n_missing > 0:
            last_frames = sequences[:, -1:, :, :].repeat(1, n_missing, 1, 1)
            sequences = torch.cat([sequences, last_frames], dim=1)

        return sequences

class VideoSlowDown(nn.Module):
    def __init__(self, repeat_factor=2):
        super().__init__()
        self.repeat_factor = repeat_factor

    def forward(self, sequences):
        N, T, V, C = sequences.shape

        # Repeat each frame by repeat_factor
        repeated_sequences = sequences.unsqueeze(2).repeat(1, 1, self.repeat_factor, 1, 1)  # Shape: (N, T, repeat_factor, V, C)
        repeated_sequences = repeated_sequences.view(N, -1, V, C)  # Flatten to (N, T * repeat_factor, V, C)

        # Take the first T frames to maintain original size
        slowed_sequences = repeated_sequences[:, :T, :, :]

        return slowed_sequences


class RandomApplyTransform(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()

        self.transform = transform
        self.p = p

    def forward(self, batch):

        return torch.stack([self.transform(img.unsqueeze(0))[0] if random.random() < self.p else img for img in batch])

def augmentations_sequence1():
    augmentations = v2.Compose([
        RandomApplyTransform(HideSequenceSegment(min_ratio=0.1, max_ratio=0.3), p=0.5),
        RandomApplyTransform(JointSubtract(joint=1), p=0.5),
        RandomApplyTransform(Rotate(axis='z', angle=30), p=0.5),
        RandomApplyTransform(Shear(), p=0.5),
        RandomApplyTransform(RandomPerspectiveBatch(), p=0.5),
        RandomApplyTransform(RandomHorizontalFlip(), p = 0.5),
        RandomApplyTransform(VideoAccelerate(), p = 0.5),
    ])
    return augmentations

def augmentations_sequence2():
    augmentations = v2.Compose([
        RandomApplyTransform(RandomTemporalCropAndPad(min_crop_length=12), p=0.5),
        RandomApplyTransform(JointSubtract(joint=1), p=0.5),
        RandomApplyTransform(Rotate(axis='z', angle=30), p=0.5),
        RandomApplyTransform(Shear(), p=0.5),
        RandomApplyTransform(RandomPerspectiveBatch(), p=0.5),
        RandomApplyTransform(RandomHorizontalFlip(), p = 0.5),
        RandomApplyTransform(VideoSlowDown(), p = 0.5)
    ])
    return augmentations