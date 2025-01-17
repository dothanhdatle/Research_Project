import os
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class SHREC22_data(Dataset):
    def __init__(self, path, T, normalize=False, skeleton_connections=None):
        """
        :param path: path to the dataset directory
        :param T: parameter deciding sequence length. All sequence with length smaller than T will be zero-padded to length T.
        :param normalize: normalize the data
        :param skeleton_connections: list of connections between 2 joints
        """

        self.path = path
        self.T = T
        self.normalize = normalize

        # skeleton joint connections
        if skeleton_connections is not None:
            self.skeleton_connections = skeleton_connections
        else:
            self. skeleton_connections = [(0, 2), (0, 6), (0, 1), (0, 16),
                                          (0, 21), (2, 3), (3, 4), (4, 5),
                                          (6, 7), (7, 8), (8, 9), (9, 10), 
                                          (1, 11), (11, 12), (12, 13), 
                                          (13, 14), (14, 15), (16, 17),
                                          (17, 18), (18, 19), (19, 20), 
                                          (21, 22), (22, 23), (23, 24), (24, 25)]
        
        self.label_map = [
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "OK",
            "MENU",
            "LEFT",
            "RIGHT",
            "CIRCLE",
            "V",
            "CROSS",
            "GRAB",
            "PINCH",
            "DENY",
            "WAVE",
            "KNOB",
        ]

        if self.normalize:
            self.mean, self.std = self.compute_normalization_stats()
            for i, sequence in enumerate(self.sequences):
                sequence = (sequence - self.mean) / (self.std + 1e-8)  # Avoid divide by 0
                self.sequences[i] = sequence
        
        
        # Create adjacency matrix with the skeleton connections
        self.adjacency_matrix = self.adjacency_matrix()

        sequences, labels = [], []
        # Each annotation row contains: "sequenceNum;GestureLabel;GSFrame;GEFrame;...;GestureLabel;GSFrame;GEFrame;"
        with open(os.path.join(self.path, 'annotations.txt'),'r') as f:
            for line in f:
                seq = line.strip().split(';')
                sequence_id = seq[0] # sequence number correspond to sequence file name
                
                frames = []
                with open(os.path.join(self.path, f"{sequence_id}.txt"), "r") as fp:
                    # Each line of sequence file contains: 'Frame Index(integer); Time_stamp(float); Joint1_x (float);
                    # Joint1_y; Joint1_z; Joint2_x; Joint2_y; Joint2_z; ...'.
                    for line in fp:
                        frame = line.split(";")[2:-1]
                        frame = (np.reshape(frame, (26, 3)).astype(np.float64))
                        frames.append(frame)
                
                frames = np.array(frames)
                for i in range(1, len(seq)-1, 3):
                    # gesture label
                    gesture_label  = seq[i]
                    # gesture start frame
                    start_frame = int(seq[i+1])
                    # gesture end frame
                    end_frame = int(seq[i+2])

                    sequences.append(frames[start_frame:end_frame+1])
                    labels.append(self.label_map.index(gesture_label))
        
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        original_len = len(sequence)
        sequence = self._preprocess(sequence)
        label = self.labels[idx]
        return dict(Sequence=sequence, Label=label, original_length = original_len)
    
    def compute_normalization_stats(self):
        """
        Compute mean and standard deviation for normalization.
        """
        all_data = []
        for seq in self.sequences:
            all_data.append(torch.from_numpy(seq))
        
        all_data = torch.cat(all_data, dim=0)
       
        mean = all_data.mean(dim=0)  
        std = all_data.std(dim=0)    
        
        return mean, std
    
    def pad_sequence(self, seq, target_length):
        seq_len = seq.shape[0]
        if seq_len >= target_length:
            return seq[:target_length]
        pad_len = target_length - seq_len
        padding = torch.zeros((pad_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)

        return torch.cat([seq, padding], dim=0)
    
    def _preprocess(self, sequence: np.ndarray):
        sequence = np.array(sequence)
        # convert to tensor
        sequence_tensor = torch.from_numpy(sequence)
        if len(sequence_tensor.shape) > len(sequence.shape):
            sequence_tensor = sequence_tensor.unsqueeze(0)
        
        sequence_tensor = self.pad_sequence(sequence_tensor, self.T)

        return sequence_tensor
    
    def adjacency_matrix(self):
        """
        Create adjacency matrix for the given skeleton connections.
        """
        num_joints = 26  # number of joints
        adjacency_matrix = np.zeros((num_joints, num_joints))
        for i, j in self.skeleton_connections:
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1 
        return adjacency_matrix


    
    
            
