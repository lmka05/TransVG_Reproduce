# ==============================================================================
# image_transforms.py — Xử lý ảnh đơn giản (giống SeqTR)
# ==============================================================================
# Chỉ 4 hàm cơ bản:
#   1. resize_image_keep_ratio() — resize giữ tỉ lệ
#   2. pad_image_to_square()     — pad về hình vuông 640×640
#   3. normalize_image()         — chia 255 về [0, 1]
#   4. create_image_mask()       — tạo mask cho Transformer
#
# So với TransVG_Reimplement (cũ): bỏ toàn bộ augmentation (crop, flip, jitter...)
# So với SeqTR: giống hệt, chỉ thêm create_image_mask()
# ==============================================================================

import numpy as np
import torch
from PIL import Image


def resize_image_keep_ratio(img, max_size):
    """
    Resize ảnh sao cho cạnh dài nhất = max_size, GIỮ NGUYÊN tỉ lệ.

    Args:
        img (np.ndarray): Ảnh gốc [H, W, 3], dtype=uint8
        max_size (int): Kích thước tối đa (640)

    Returns:
        resized_img (np.ndarray): Ảnh đã resize [new_H, new_W, 3]
        scale (float): Tỉ lệ scale

    Ví dụ:
        Ảnh 800×600, max_size=640
        scale = 640 / max(800, 600) = 0.8
        new_size = (640, 480)
    """
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h))
    resized_img = np.array(pil_img)

    return resized_img, scale


def pad_image_to_square(img, target_size, pad_value=0):
    """
    Pad ảnh về hình vuông target_size × target_size.
    Padding ở bên PHẢI và bên DƯỚI (giống SeqTR).

    Args:
        img (np.ndarray): Ảnh đã resize [H, W, 3]
        target_size (int): Kích thước đích (640)
        pad_value (int): Giá trị pixel padding (0 = đen)

    Returns:
        padded_img (np.ndarray): [target_size, target_size, 3]

    Ví dụ:
        ┌──────────────┐
        │  Ảnh gốc     │ 480px
        │  640 × 480   │
        ├──────────────┤
        │  Padding (0) │ 160px
        └──────────────┘
    """
    h, w = img.shape[:2]
    padded = np.full((target_size, target_size, 3), pad_value, dtype=img.dtype)
    padded[:h, :w, :] = img
    return padded


def normalize_image(img):
    """
    Normalize pixel values: [0, 255] → [0, 1]
    Giống SeqTR (đơn giản chia 255, không dùng ImageNet mean/std).
    """
    return img.astype(np.float32) / 255.0


def image_to_tensor(img):
    """
    Chuyển numpy [H, W, C] → tensor [C, H, W].
    """
    img = np.transpose(img, (2, 0, 1))    # [H,W,C] → [C,H,W]
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img)


def create_image_mask(resized_h, resized_w, target_size):
    """
    Tạo mask cho Transformer: 0 = ảnh thật, 1 = padding.

    TransVG cần mask này để Transformer KHÔNG attend vào vùng padding.
    (SeqTR không cần mask vì dùng CNN backbone — CNN tự xử lý padding)

    Args:
        resized_h (int): Chiều cao ảnh SAU resize (trước padding)
        resized_w (int): Chiều rộng ảnh SAU resize (trước padding)
        target_size (int): Kích thước ảnh SAU padding (640)

    Returns:
        mask (Tensor): [target_size, target_size], dtype=bool
            False = pixel ảnh thật (Transformer sẽ attend)
            True  = pixel padding (Transformer sẽ ignore)

    Ví dụ (ảnh 640×480 pad thành 640×640):
        ┌──────────────────┐
        │  False  False  F │ ← ảnh thật (480 dòng)
        │  False  False  F │
        ├──────────────────┤
        │  True   True   T │ ← padding (160 dòng)
        └──────────────────┘
    """
    mask = torch.ones(target_size, target_size, dtype=torch.bool)   # Tất cả = True (padding)
    mask[:resized_h, :resized_w] = False                            # Vùng ảnh = False
    return mask


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test image_transforms ===\n")

    # Tạo ảnh giả 800×600
    img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    print(f"Original: {img.shape}")  # (600, 800, 3)

    # Resize
    img_resized, scale = resize_image_keep_ratio(img, 640)
    print(f"Resized:  {img_resized.shape}, scale={scale:.2f}")  # (480, 640, 3), 0.80

    # Pad
    img_padded = pad_image_to_square(img_resized, 640)
    print(f"Padded:   {img_padded.shape}")  # (640, 640, 3)

    # Normalize
    img_norm = normalize_image(img_padded)
    print(f"Normed:   range=[{img_norm.min():.2f}, {img_norm.max():.2f}]")  # [0, 1]

    # To tensor
    img_tensor = image_to_tensor(img_norm)
    print(f"Tensor:   {img_tensor.shape}")  # [3, 640, 640]

    # Mask
    rh, rw = img_resized.shape[:2]
    mask = create_image_mask(rh, rw, 640)
    print(f"Mask:     {mask.shape}")  # [640, 640]
    print(f"Mask real pixels: {(~mask).sum().item()}")    # 480*640
    print(f"Mask padding:     {mask.sum().item()}")       # 160*640

    print("\n✅ image_transforms test passed!")
