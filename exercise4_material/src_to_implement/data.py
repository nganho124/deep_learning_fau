from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from PIL import Image


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self, data, mode):

        self.data = data
        self.mode = mode

        transform_list = [tv.transforms.ToPILImage()]

        if mode == "train":

            # Add augmentations for training
            transform_list.extend([
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomRotation(degrees=10),
                tv.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
            
        # Common transforms for both train and val
        transform_list.extend([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])
            
        self._transform = tv.transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        # Get row
        row = self.data.iloc[index]
        
        # Read image using PIL directly
        image = Image.open(row['filename'])
        
        # Convert to numpy for gray2rgb
        img_array = np.array(image)
        
        # Ensure RGB
        if img_array.ndim == 2:
            img_array = gray2rgb(img_array)
        
        # Apply transforms
        img_tensor = self._transform(img_array)
        
        # Labels
        labels = torch.tensor([row['crack'], row['inactive']], dtype=torch.float32)
        
        return img_tensor, labels