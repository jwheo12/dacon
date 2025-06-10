
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuration
CFG = {
    'IMG_SIZE': 384,
    'BATCH_SIZE': 32,
    'EPOCHS': 100,
    'LEARNING_RATE': 1e-4,
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        if is_test:
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

def get_transforms():
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop((CFG['IMG_SIZE'], CFG['IMG_SIZE']), scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(15),
        v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandAugment(),
        v2.ColorJitter(0.2, 0.2, 0.2),
        v2.RandomAutocontrast(),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        v2.RandomErasing(p=0.25),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE']), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def get_loaders(train_root='train', test_root='test'):
    # Seed before splitting
    seed_everything(CFG['SEED'])

    # full dataset for stratification
    full_dataset = CustomImageDataset(train_root, transform=None)
    targets_all = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes

    train_idx, val_idx = train_test_split(
        range(len(targets_all)),
        test_size=0.2,
        stratify=targets_all,
        random_state=CFG['SEED']
    )
    train_transform, val_transform = get_transforms()

    train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
    val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)

    # Weighted sampler for class imbalance
    cls_counts = np.bincount([full_dataset.samples[i][1] for i in train_idx])
    class_weights = 1.0 / cls_counts
    sample_weights = [class_weights[label] for label in [full_dataset.samples[i][1] for i in train_idx]]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG['BATCH_SIZE'],
        sampler=weighted_sampler,
        num_workers=16,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Test loader
    test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names
