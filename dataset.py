import torch
import cv2
import torchvision
from torchvision import transforms
import numpy
import os
from torch.utils.data import Dataset, DataLoader


class EventData(Dataset):

    def __init__(self, root, seq, event_dir, frame_dir, transform=None):
        self.root = root
        self.event_dir = root+seq+event_dir
        self.event_tensors = []
        for f in sorted(os.listdir(self.event_dir)):
            if f.endswith('npy'):
                self.event_tensors.append(f)
        if len(self.event_tensors) % 2 != 0:
            self.event_tensors.pop()

        self.frame_dir = root+seq+frame_dir
        self.frames = []
        for f in sorted(os.listdir(self.frame_dir)):
            if f.endswith('png'):
                self.frames.append(f)
        if len(self.event_tensors) % 2 != 0:
            self.frames.pop()

        self.transform = transform


    def __len__(self):
        return len(self.event_tensors)


    def __getitem__(self, index):
        event_name = os.path.join(self.event_dir, self.event_tensors[index])  
        frame_name = os.path.join(self.frame_dir, self.frames[index])  
        event = numpy.load(event_name)
        event_tensor = torch.tensor(event)
        frame = cv2.imread(frame_name)
        frame_tensor = torch.tensor(numpy.transpose(frame, (2,0,1)))
        frame_tensor = frame_tensor.type(torch.FloatTensor)

        # Normalize event tensor to mean=0, std=1
        mean = torch.mean(event_tensor)
        std = torch.std(event_tensor)
        event_tensor = (event_tensor - mean)/std

        # Padding
        padx = torch.zeros((5,2,240))
        pady = torch.zeros((3,2,240))
        event_t = torch.cat((padx, event_tensor, padx), 1)
        frame_t = torch.cat((pady, frame_tensor, pady), 1)

        return (event_t, frame_t)
