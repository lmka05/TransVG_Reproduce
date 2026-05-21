# ==============================================================================
# datasets/__init__.py — Tạo DataLoader nhanh
# ==============================================================================

from .dataset import RefCOCODataset
from .dataloader import build_dataloader


def build_train_loader(config):
    """Tạo train DataLoader."""
    dataset = RefCOCODataset(config.ann_file, config.img_dir, 'train', config)
    return build_dataloader(dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)


def build_val_loader(config, split='val'):
    """Tạo val/test DataLoader."""
    dataset = RefCOCODataset(config.ann_file, config.img_dir, split, config)
    return build_dataloader(dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)
