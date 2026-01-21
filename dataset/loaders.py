from torch.utils.data import DataLoader
from data.dataset import INaturalistDataset
from data.transforms import get_transforms


def get_train_val_loaders(
    root_dir,
    train_split="train_split",
    val_split="val_split",
    batch_size=32,
    num_workers=4,
    augment=True,
    image_size=224
):
    train_dataset = INaturalistDataset(
        root_dir=root_dir,
        split=train_split,
        transform=get_transforms(train=True, augment=augment, image_size=image_size)
    )

    val_dataset = INaturalistDataset(
        root_dir=root_dir,
        split=val_split,
        transform=get_transforms(train=False, augment=False, image_size=image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
