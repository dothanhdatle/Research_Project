import random

import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2


class Rotate:
    def __init__(self, axis=None, angle=None):
        self.axis = axis
        self.angle = angle

    def __call__(self, sequence):
        N, T, V, C = sequence.shape
        device = sequence.device
        temp = sequence.clone()

        axis = self.axis if self.axis is not None else random.randint(0, 2)

        # Determine angle
        if self.angle is not None:
            if isinstance(self.angle, list):
                angle_next = random.uniform(self.angle[0] - self.angle[1], self.angle[0] + self.angle[1])
            else:
                angle_next = self.angle
        else:
            angle_next = random.uniform(-30, 30)

        # Convert angle to radians
        angle_rad = torch.deg2rad(torch.tensor(angle_next, dtype=sequence.dtype, device=device))
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

        # Define rotation matrices for the specified axis
        if axis == 0:  # Rotate around X-axis
            R = torch.tensor([[1, 0, 0],
                              [0, cos_a, sin_a],
                              [0, -sin_a, cos_a]], device=device, dtype=sequence.dtype)

        elif axis == 1:  # Rotate around Y-axis
            R = torch.tensor([[cos_a, 0, -sin_a],
                              [0, 1, 0],
                              [sin_a, 0, cos_a]], device=device, dtype=sequence.dtype)

        elif axis == 2:  # Rotate around Z-axis
            R = torch.tensor([[cos_a, sin_a, 0],
                              [-sin_a, cos_a, 0],
                              [0, 0, 1]], device=device, dtype=sequence.dtype)

        # Apply the rotation using matrix multiplication across the batch
        temp = temp.view(-1, C)
        temp = torch.matmul(temp, R.T)
        temp = temp.view(N, T, V, C)

        return temp

class Shear:
    def __init__(self, s1=None, s2=None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, sequence):
        # Create a clone of the tensor to avoid modifying the original data
        temp = sequence.clone()
        device = sequence.device

        # Generate random shear values if not provided
        if self.s1 is not None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        
        if self.s2 is not None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

        # Construct the shear matrix
        shear_matrix = torch.tensor([
            [1, s1_list[0], s2_list[0]],  
            [s1_list[1], 1, s2_list[1]],  
            [s1_list[2], s2_list[2], 1]   
        ], device=device, dtype=sequence.dtype) 

        # Apply shear transformation
        temp = torch.einsum('ntvc,cd->ntvd', temp, shear_matrix.T)

        return temp
    
class RandomHorizontalFlip:
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sequence):
        temp = sequence.clone()

        # Apply flip with probability p
        if random.random() < self.p:
            # Flip along the X-axis
            temp[..., 0] *= -1 

        return temp

class GaussianNoise:
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std


    def __call__(self, sequence):

        noise = torch.randn_like(sequence) * self.std
        sequence_noise = sequence + noise
        sequence_noise[sequence == 0] = 0

        return sequence + noise
    
class GaussianBlurConv(nn.Module):
    def __init__(self, channels = 3, kernel_length = 15, sigma_range = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel_length = kernel_length
        self.sigma_range = sigma_range
        radius = int(kernel_length / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, sequence):
        device = sequence.device
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0).float()
        # kernel =  kernel.float()
        kernel = kernel.repeat(self.channels, 1, 1, 1).to(device) # (C,1,1,kernel_size)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        temp = sequence.clone()
        if prob < 0.5:
            temp = temp.permute(0,3,2,1) # N,C,V,T
            temp = F.conv2d(temp, self.weight, padding=(0, int((self.kernel_length - 1) / 2 )), groups=self.channels)
            temp = temp.permute(0,3,2,1) # N,T,V,C

        return temp

class MotionBlurConv(nn.Module):
    def __init__(self, channels=3, kernel_length=15, direction='horizontal'):
        """
        Apply motion blur to skeleton-based data.
        
        :param channels: Number of channels
        :param kernel_length: The length of the motion blur kernel.
        :param direction: The direction of the blur ('horizontal' or 'vertical').
        """
        super(MotionBlurConv, self).__init__()
        self.channels = channels
        self.kernel_length = kernel_length
        self.direction = direction
        
        # Create a motion blur kernel based on the direction and kernel size
        self.kernel = self.create_motion_blur_kernel(kernel_length, direction)

    def create_motion_blur_kernel(self, kernel_length, direction):
        """
        Create a motion blur kernel of given length and direction.

        :param kernel_length: The length of the motion blur kernel.
        :param direction: The direction of the blur ('horizontal' or 'vertical').
        :return: The motion blur kernel.
        """
        # Create the kernel values: Uniformly distributed over the kernel length
        kernel = np.ones(kernel_length) / kernel_length
        
        # Convert to a PyTorch tensor and reshape based on the direction
        if direction == 'horizontal':
            # Horizontal motion blur (applies across the temporal axis)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, kernel_length)
        elif direction == 'vertical':
            # Vertical motion blur (applies across the spatial axis or joints)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_length, 1)
        else:
            raise ValueError("Direction must be either 'horizontal' or 'vertical'.")

        kernel = kernel.double()  # Convert to double precision
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # Replicate for each channel
        return kernel

    def __call__(self, sequence):
        """
        Apply motion blur to the input sequence of skeleton data.

        :param sequence: Input sequence of shape (N, C, V, T), where N is batch size, 
                         C is channels, V is the number of joints, T is time steps.
        :return: The motion-blurred sequence.
        """
        # Convert the input to a torch tensor if it's a numpy array
        sequence = torch.from_numpy(sequence)

        # Apply the motion blur to the sequence with convolution
        prob = np.random.random_sample()
        if prob < 0.5:
            sequence = sequence.permute(0, 3, 2, 1)  # N, T, V, C -> N, C, V, T (for conv2d)
            sequence = F.conv2d(sequence, self.kernel, padding=(0, int((self.kernel_length - 1) / 2)), groups=self.channels)
            sequence = sequence.permute(0, 3, 2, 1)  # Revert to N, T, V, C
        
        return sequence


class RandomApplyTransform(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()

        self.transform = transform
        self.p = p

    def forward(self, batch):

        return torch.stack([self.transform(img.unsqueeze(0))[0] if random.random() < self.p else img for img in batch])
    
def augmentations_sequence():
    augmentations = v2.Compose([
        RandomApplyTransform(Rotate(angle=[30,30]), p=0.5),
        RandomApplyTransform(Shear(), p=0.5),
        RandomApplyTransform(RandomHorizontalFlip(p=1), p = 0.5),
        RandomApplyTransform(GaussianBlurConv(channels=3,kernel_length=15,sigma_range=[0.1,2]), p=0.5),
    ])
    return augmentations