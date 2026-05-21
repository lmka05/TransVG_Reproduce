# ==============================================================================
# misc.py — Tiện ích cho TransVG
# ==============================================================================
# Chỉ 2 thành phần cốt lõi:
#   1. NestedTensor — gói tensor + mask thành 1 object
#   2. collate_fn   — gom batch cho DataLoader
#
# So với TransVG_Reimplement (cũ): bỏ SmoothedValue + MetricLogger
# ==============================================================================

import torch


class NestedTensor:
    """
    Gói 1 tensor + 1 mask thành 1 object.

    Tại sao cần?
        Transformer cần biết vùng nào là padding để ignore trong attention.
        NestedTensor gói (data + mask) lại → truyền 1 object thay vì 2.

    Dùng cho:
        - Image: tensors=[B,3,640,640], mask=[B,640,640]
        - Text:  tensors=[B,17], mask=[B,17]

    Ví dụ:
        img_data = NestedTensor(images, image_masks)
        img_data.to('cuda')  # chuyển cả tensors và mask sang GPU
        imgs, masks = img_data.decompose()  # tách ra
    """

    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """Chuyển cả tensors và mask sang device (GPU/CPU)."""
        tensors = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        """Tách ra tensors và mask."""
        return self.tensors, self.mask


def collate_fn(batch):
    """
    Gom nhiều samples thành 1 batch cho DataLoader.

    Mỗi sample từ dataset: (img, img_mask, word_id, word_mask, bbox)
    Output batch: (NestedTensor_img, NestedTensor_text, bbox_batch)

    Args:
        batch: list of (img, img_mask, word_id, word_mask, bbox)

    Returns:
        img_data:  NestedTensor(imgs [B,3,640,640], masks [B,640,640])
        text_data: NestedTensor(word_ids [B,17], word_masks [B,17])
        bbox:      Tensor [B, 4]
    """
    imgs, img_masks, word_ids, word_masks, bboxes = zip(*batch)

    # Stack tensors
    imgs = torch.stack(imgs, dim=0)           # [B, 3, 640, 640]
    img_masks = torch.stack(img_masks, dim=0) # [B, 640, 640]
    word_ids = torch.stack(word_ids, dim=0)   # [B, 17]
    word_masks = torch.stack(word_masks, dim=0)  # [B, 17]
    bboxes = torch.stack(bboxes, dim=0)       # [B, 4]

    # Gói thành NestedTensor
    img_data = NestedTensor(imgs, img_masks)
    text_data = NestedTensor(word_ids, word_masks)

    return img_data, text_data, bboxes


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test misc.py ===\n")

    # Test NestedTensor
    imgs = torch.randn(2, 3, 640, 640)
    masks = torch.zeros(2, 640, 640, dtype=torch.bool)
    nt = NestedTensor(imgs, masks)

    t, m = nt.decompose()
    print(f"NestedTensor: tensors={t.shape}, mask={m.shape}")

    # Test collate_fn
    batch = [
        (torch.randn(3, 640, 640), torch.zeros(640, 640, dtype=torch.bool),
         torch.randint(0, 100, (17,)), torch.ones(17, dtype=torch.long),
         torch.tensor([0.5, 0.5, 0.3, 0.4])),
        (torch.randn(3, 640, 640), torch.zeros(640, 640, dtype=torch.bool),
         torch.randint(0, 100, (17,)), torch.ones(17, dtype=torch.long),
         torch.tensor([0.3, 0.3, 0.2, 0.2])),
    ]
    img_data, text_data, bbox = collate_fn(batch)
    print(f"\nCollate output:")
    print(f"  img_data:  tensors={img_data.tensors.shape}, mask={img_data.mask.shape}")
    print(f"  text_data: tensors={text_data.tensors.shape}, mask={text_data.mask.shape}")
    print(f"  bbox:      {bbox.shape}")

    print("\n✅ misc.py test passed!")
