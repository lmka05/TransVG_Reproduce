# ==============================================================================
# datasets/dataloader.py — DataLoader cho TransVG
# ==============================================================================
# Tạo DataLoader từ dataset, dùng collate_fn để gom batch thành NestedTensor.
# ==============================================================================

from torch.utils.data import DataLoader
from utils.misc import collate_fn


def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """
    Tạo DataLoader từ dataset.

    Args:
        dataset: RefCOCODataset
        batch_size (int): Số sample mỗi batch
        shuffle (bool): Xáo trộn (True cho train, False cho val/test)
        num_workers (int): Số process song song load data

    Returns:
        DataLoader — mỗi batch gồm (img_data, text_data, bbox):
            img_data:  NestedTensor(imgs [B,3,640,640], masks [B,640,640])
            text_data: NestedTensor(word_ids [B,17], word_masks [B,17])
            bbox:      Tensor [B, 4]
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=(shuffle == True),  # Drop batch cuối khi train
        persistent_workers=(num_workers > 0)
    )
