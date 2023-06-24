# Import the necessary libraries
import random

from pathlib import Path
import PIL.Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("tableau-colorblind10")

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

# Define a torch dataset to load the images
class FaceDataset(Dataset):
    def __init__(self, root_dir):
        # Call the superclass constructor
        super(FaceDataset, self).__init__()

        # Define the basic transforms i.e. convert the array to PIL Image
        # Then tensorize it and normalize it.
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        # Define the root directory where all the images are stored
        self.root_dir = root_dir
        
        # Figure out all the classes from the root dir folders
        classes = [x.name for x in Path(root_dir).glob("*") if (x.is_dir() and x.name != ".ipynb_checkpoints")]
        classes = sorted(classes)
        self.class_map = {cl: idx for idx, cl in enumerate(classes)}
        
        # Get a list of all the file names
        files = [x for x in Path(root_dir).glob("**/*") if (x.is_file() and x.parent.name != ".ipynb_checkpoints")]
        random.seed(42); random.shuffle(files)
        self.image_paths = files
        
    def __getitem__(self, index):
        
        # Get the path of the image
        pth = self.image_paths[index]
        
        # Get the image label 
        label = self.class_map[pth.parent.name]
        label = torch.tensor(label, dtype=torch.long)
        
        # Read the image pixel values
        sample = np.array(PIL.Image.open(pth))
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label
    
    def __len__(self):
        return len(self.image_paths)
    
    def plot_img(self, i, a):
        # Given the image and axis object on which to plot, open the image and render it on the axis object
        a.imshow(PIL.Image.open(i))
        a.set_xticks([]); a.set_yticks([])
        cl = i.parent.name
        a.set_title(f"{cl}; [{self.class_map[cl]}]")
    
    def visualize_dataset(self):
        # Visualize some samples from the dataset
        samples = random.sample(self.image_paths, 16)
        fig, ax = plt.subplots(4, 4, figsize = (10,10))
        for s, a in zip(samples, ax.flat): self.plot_img(s, a)

if __name__ == "__main__":
    DATASET_PATH = "/home/vinayak/Face-Applications/datasets/celebrity_images/"
    BATCH_SIZE = 16
    dset = FaceDataset(DATASET_PATH)
    dloader = DataLoader(dset, batch_size = BATCH_SIZE, shuffle=True)