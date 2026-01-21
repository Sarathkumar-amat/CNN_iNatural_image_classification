import os
from PIL import Image
from torch.utils.data import Dataset

class INaturalistDataset(Dataset):
    def __init__(self, root_dir, split="train",transform=None):
        assert split in ["train", "validation"], \
            "split must be 'train' or 'validation'"
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(root_dir, split)

        self.image_paths = []
        self.labels = []

        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
