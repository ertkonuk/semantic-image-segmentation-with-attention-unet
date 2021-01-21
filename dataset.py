# . . import torch Dataset class
from torch.utils.data import Dataset
import torchvision.transforms as T
# . . import the utilities
import utils
# . . numpy
import numpy as np
# .. python utils
import os
#import imageio
import cv2

# . .  the dataset class inherits from torch.utils.data.Dataset
class TGSSaltDataset(Dataset):

    # . . the constructor 
    def __init__(self, path, files, transform=None, imgsize=None, mask=True, grayscale=False):
        self.path = path
        self.files = files
        self.transform = transform
        self.mask = mask
        self.grayscale = grayscale
        self.imgsize= imgsize
    # . . get one item from the dataset         
    def __getitem__(self, index):

        # . . get the file ID
        file_id  = self.files[index]
        
        # . . get the image directory
        img_dir  = os.path.join(self.path, 'images')
        # . . get the file name for the image
        img_file = os.path.join(img_dir, file_id + '.png')

        # . . read the image
        image = cv2.imread(img_file)
        
        # . . if resize: if a new image size is given
        if self.imgsize is not None:
            image = cv2.resize(image, self.imgsize)
        
        # . . convert to grayscale
        #if self.grayscale:
        #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # . . if the image has mask
        if self.mask:
            # . . get the mask directory
            mask_dir  = os.path.join(self.path, 'masks')
            # . . get the file name for the mask
            mask_file = os.path.join(mask_dir, file_id + '.png')
            # . . read the mask
            mask = cv2.imread(mask_file)
            # . . if a new image size is given
            if self.imgsize is not None:
                mask = cv2.resize(mask, self.imgsize)

            ## . . the mask is always grayscale
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # . . convert to tensor
        to_tensor = T.ToTensor()
        to_gray   = T.Grayscale()

        if self.transform is not None:
            # . . apply user-defined transforms 
            if self.mask:
                transformed = self.transform(image=image, mask=mask)            
                image = transformed['image']
                mask  = transformed['mask']

            else:
                transformed = self.transform(image)
                image = transformed['image']

        
        if self.mask:
            # . . convert to tensor
            image = to_tensor(image)
            mask  = to_tensor(mask)  
            if self.grayscale:
                image = to_gray(image)          
                mask  = to_gray(mask) 
            return image, mask
        else: 
            # . . convert to tensor
            image = to_tensor(image)
            if self.grayscale:
                image = to_gray(image)
            return image

    def __len__(self):
        return len(self.files)