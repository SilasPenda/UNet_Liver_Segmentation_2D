import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import  torchvision.transforms.functional as TF
from torchvision import transforms

from src.utils import get_config


class LoadTransformDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and corresponding label paths
        image_path = os.path.join(self.images_dir, self.images[idx])
        rgb_mask_path = os.path.join(self.labels_dir, self.labels[idx])
    
        image = cv2.imread(image_path)
        mask = cv2.imread(rgb_mask_path, cv2.IMREAD_GRAYSCALE)

        # image = cv2.resize(image, (128, 128))
        # mask = cv2.resize(mask, (128, 128))

        mask[mask <= 0.5] = 0  # Mask intensities
        mask[(mask > 0.5) & (mask <= 1.5)] = 1
        mask[mask > 1.5] = 2

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
 
        image = TF.to_tensor(image)
        mask = torch.tensor(mask, dtype=torch.long) 
        
        # mask = TF.to_tensor(mask).squeeze(0)  # Remove channel dimension


        return image, mask

    
def pad_tensor(tensor, max_height, max_width):
    # Get the current shape of the tensor
    if len(tensor.shape) == 3:
        _, height, width = tensor.shape  # First dimension is the channel
    else:
        height, width = tensor.shape

    # Calculate the padding required
    pad_height = max_height - height
    pad_width = max_width - width

    # Pad the tensor (pad only height and width, not channels)
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height))  # (left, right, top, bottom)
    
    return padded_tensor

def pad_collate_fn(batch):
    # Get the maximum height and width for the batch
    max_height = max([item[0].shape[-2] for item in batch])  # item[0] is the image
    max_width = max([item[0].shape[-1] for item in batch])

    # Apply padding to each image and label
    padded_images = [pad_tensor(img, max_height, max_width) for img, _ in batch]
    padded_masks = [pad_tensor(mask, max_height, max_width) for _, mask in batch]

    # Stack the images and labels into batches
    batch_images = torch.stack(padded_images)
    batch_masks = torch.stack(padded_masks)

    return batch_images, batch_masks



def get_data_loaders():   
    # Define training transformations
    # train_transforms = transforms.Compose([
    #     transforms.RandomRotation(35),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.1),
    #     # transforms.ToTensor(),
    # ])


    # Loading config and setting up paths (same as before)
    config = get_config(config_filepath=os.path.join(os.getcwd(), 'config.yaml'))
    train_dir = config.get('train_dir', None)
    val_dir = config.get('val_dir', None)

    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")


    # Create dataset and DataLoader
    train_dataset = LoadTransformDataset(train_images_dir, train_labels_dir)
    val_dataset = LoadTransformDataset(val_images_dir, val_labels_dir)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False, pin_memory=True, num_workers=2)

    # batch_size = 2
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)

    return train_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

    for images, labels in val_loader:
        # Process the first image
        print(images.shape)
        image = images[0].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
        image = (image * 255).astype(np.uint8)  # Rescale and convert to uint8
        # print(f"Image min: {image.min()}, max: {image.max()}")

        plt.subplot(1, 2, 1)  # Create a 1x2 grid, use the first subplot
        plt.imshow(image)  # Image is in (H, W, C) format after permute
        # plt.title("Image")
        print(np.unique(labels[0]))
        # print(labels[0].shape)

        # Process the first label (remove the channel dimension)
        label = labels[0].cpu().squeeze().numpy()

        plt.subplot(1, 2, 2)  # Use the second subplot
        plt.imshow(label, cmap="gray")  # Use cmap="gray" for single-channel images
        plt.title("Label")

        # Save the figure
        plt.savefig("save.png")
        # plt.show
        break