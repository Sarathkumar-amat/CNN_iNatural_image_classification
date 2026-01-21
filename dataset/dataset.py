import os
from PIL import Image
from torch.utils.data import Dataset


class INaturalistDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to iNaturalist root directory
            split (str): 'train' or 'validation'
            transform (callable): torchvision transforms
        """
        assert split in ["train", "validation"], \
            "split must be 'train' or 'validation'"

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(root_dir, split)

        self.image_paths = []
        self.labels = []

        # Sort classes for deterministic label assignment
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        # Collect image paths
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            label = self.class_to_idx[class_name]

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(
                        os.path.join(class_dir, fname)
                    )
                    self.labels.append(label)

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
