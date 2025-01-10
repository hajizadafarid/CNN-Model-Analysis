import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple
import numpy as np


def set_all_seeds(seed):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch
    torch.cuda.manual_seed(seed)     # PyTorch CUDA
    torch.cuda.manual_seed_all(seed) # PyTorch CUDA for all GPUs
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False     # Disable CUDA benchmark mode
    print('done')


class CategoryImageDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 file_list: List[str],  # Now takes a specific list of files
                 transform=None,
                 target_size=(224, 224)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get class names and create class-to-index mapping
        self.classes = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Use the provided file list instead of scanning directory
        self.images = file_list
        self.targets = [self.class_to_idx[os.path.basename(os.path.dirname(img_path))] 
                       for img_path in file_list]

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Load and transform image at given index"""
        img_path = self.images[idx]
        target = self.targets[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = image * 2 - 1
        return image, target


def split_train_eval_files(root_dir: str, 
                          train_samples_per_class: int,
                          eval_samples_per_class: int) -> Tuple[List[str], List[str]]:
    """
    Split available images into train and eval sets ensuring no overlap
    """
    train_files = []
    eval_files = []
    
    # Process each class directory
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get all valid image files
        all_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Ensure we have enough images
        total_needed = train_samples_per_class + eval_samples_per_class
        if len(all_images) < total_needed:
            raise ValueError(
                f"Class {class_name} has only {len(all_images)} images, "
                f"but {total_needed} are needed ({train_samples_per_class} train + "
                f"{eval_samples_per_class} eval)"
            )
        
        # Randomly shuffle and split
        random.shuffle(all_images)
        train_files.extend(all_images[:train_samples_per_class])
        eval_files.extend(all_images[train_samples_per_class:train_samples_per_class + eval_samples_per_class])
    
    return train_files, eval_files


def create_train_eval_dataloaders(
    root_dir: str,
    train_samples_per_class: int = 15,
    eval_samples_per_class: int = 15,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create separate train and eval dataloaders with fixed samples per class,
    ensuring no overlap between train and eval sets.

    Args:
        root_dir: Directory containing class subdirectories
        train_samples_per_class: Number of training samples per class
        eval_samples_per_class: Number of evaluation samples per class
        batch_size: Batch size for both loaders

    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    # First split the files into train and eval sets
    train_files, eval_files = split_train_eval_files(
        root_dir, train_samples_per_class, eval_samples_per_class
    )

    # Training transforms with augmentation
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    # ])

    # # Evaluation transforms without augmentation
    # eval_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    # ])'

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    # Create datasets with specific file lists
    train_dataset = CategoryImageDataset(
        root_dir=root_dir,
        file_list=train_files,
        transform=train_transform
    )

    eval_dataset = CategoryImageDataset(
        root_dir=root_dir,
        file_list=eval_files,
        transform=eval_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, eval_loader