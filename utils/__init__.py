# ==============================================================================
# utils/__init__.py — Export các hàm thường dùng
# ==============================================================================

from .misc import NestedTensor, collate_fn
from .box_utils import xywh2xyxy, xyxy2xywh, bbox_iou, generalized_box_iou
from .text_transforms import TextTransform
from .image_transforms import (
    resize_image_keep_ratio,
    pad_image_to_square,
    normalize_image,        # [CŨ] dùng cho SeqTR
    normalize_imagenet,     # [MỚI] dùng cho TransVG (ResNet pretrained)
    image_to_tensor,
    create_image_mask,
)
