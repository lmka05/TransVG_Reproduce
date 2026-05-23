# ==============================================================================
# datasets/dataset.py — RefCOCO Dataset cho TransVG
# ==============================================================================
# Xử lý ảnh:
#   1. Load ảnh → numpy
#   2. Resize giữ tỉ lệ (cạnh dài = 640)
#   3. Pad về 640×640
#   4. Normalize theo ImageNet mean/std (bắt buộc vì fine-tune ResNet pretrained)
#   5. Tạo image mask cho Transformer
#   6. BERT tokenize câu
#   7. Chuyển bbox về normalized xywh
#
# So với SeqTR:
#   Giống: resize + pad + normalize
#   Khác:  thêm image_mask (Transformer cần), bbox = normalized xywh (không phải pixel xyxy)
# ==============================================================================

import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.image_transforms import (
    resize_image_keep_ratio,
    pad_image_to_square,
    normalize_image,     # [CŨ] dùng cho SeqTR, ResNet đóng băng
    normalize_imagenet,  # [MỚI] dùng cho TransVG, fine-tune ResNet pretrained
    image_to_tensor,
    create_image_mask,
)
from utils.text_transforms import TextTransform


class RefCOCODataset(Dataset):
    """
    Dataset cho RefCOCO — mỗi sample gồm:
        - img:       Tensor [3, 640, 640]
        - img_mask:  Tensor [640, 640] — True=padding, False=ảnh thật
        - word_id:   Tensor [17] — BERT token IDs
        - word_mask: Tensor [17] — 1=token thật, 0=padding
        - bbox:      Tensor [4] — normalized (x_c, y_c, w, h) ∈ [0,1]
    """

    def __init__(self, ann_file, img_dir, split, config):
        """
        Args:
            ann_file (str): Đường dẫn tới instances.json
            img_dir (str): Đường dẫn tới thư mục ảnh COCO
            split (str): 'train', 'val', 'testA', 'testB'
            config: Config object
        """
        super().__init__()

        self.img_dir = img_dir
        self.split = split
        self.imsize = config.imsize  # 640

        # Load annotations
        anns_all = json.load(open(ann_file, 'r'))
        self.anns = anns_all[split]
        print(f"[{split}] Loaded {len(self.anns)} samples")

        # Text transform (BERT tokenizer)
        self.text_transform = TextTransform(
            bert_model=config.bert_model,
            max_query_len=config.max_query_len
        )

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        """
        Lấy 1 sample.

        Ảnh:  load → resize giữ tỉ lệ → pad 640×640 → /255 → tensor
        Text: BERT tokenize → (word_ids, attention_mask)
        Bbox: scale theo resize → normalize /640 → xywh format
        """
        ann = self.anns[index]

        # ==================================================================
        # 1. LOAD ẢNH
        # ==================================================================
        img_path = os.path.join(
            self.img_dir,
            "COCO_train2014_%012d.jpg" % ann['image_id']
        )

        with Image.open(img_path) as pil_img :
            img = np.array(pil_img.convert('RGB'))


        # ==================================================================
        # 2. RESIZE + PAD + NORMALIZE (giống SeqTR)
        # ==================================================================
        img, scale = resize_image_keep_ratio(img, self.imsize)
        resized_h, resized_w = img.shape[:2]

        img = pad_image_to_square(img, self.imsize)
        # [CŨ] img = normalize_image(img)    # /255 → [0, 1]  (dùng cho SeqTR)
        # [MỚI] Normalize theo ImageNet mean/std (bắt buộc cho ResNet pretrained)
        img = normalize_imagenet(img)          # ~[-2, +2]
        img = image_to_tensor(img)             # [3, 640, 640]

        # ==================================================================
        # 3. TẠO IMAGE MASK (chỉ TransVG cần, SeqTR không cần)
        # ==================================================================
        img_mask = create_image_mask(resized_h, resized_w, self.imsize)
        # img_mask: [640, 640], True=padding, False=ảnh thật

        # ==================================================================
        # 4. XỬ LÝ TEXT (BERT tokenize)
        # ==================================================================
        expressions = ann['expressions']
        if self.split == 'train':
            expression = random.choice(expressions)  # Random 1 câu (augmentation)
        else:
            expression = expressions[0]  # Luôn câu đầu (consistent)

        input_ids, attention_mask = self.text_transform(expression)
        word_id = torch.tensor(input_ids, dtype=torch.long)       # [17]
        word_mask = torch.tensor(attention_mask, dtype=torch.long) # [17]

        # ==================================================================
        # 5. XỬ LÝ BBOX
        # ==================================================================
        # Annotation bbox: [x, y, w, h] pixel (COCO format)
        x, y, w, h = ann['bbox']

        # Scale theo resize
        x1 = x * scale
        y1 = y * scale
        x2 = (x + w) * scale
        y2 = (y + h) * scale

        # Clip để không vượt biên ảnh
        x1 = np.clip(x1, 0, resized_w - 1)
        y1 = np.clip(y1, 0, resized_h - 1)
        x2 = np.clip(x2, 1, resized_w)
        y2 = np.clip(y2, 1, resized_h)

        # Chuyển sang center format + normalize về [0, 1]
        # TransVG output = normalized xywh → target cũng phải cùng format
        cx = (x1 + x2) / 2.0 / self.imsize
        cy = (y1 + y2) / 2.0 / self.imsize
        bw = (x2 - x1) / self.imsize
        bh = (y2 - y1) / self.imsize

        bbox = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)  # [4]

        return img, img_mask, word_id, word_mask, bbox


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=== Test RefCOCODataset ===\n")

    # Tạo dataset (cần có file annotations + ảnh)
    try:
        dataset = RefCOCODataset(
            Config.ann_file, Config.img_dir, 'train', Config
        )
        img, img_mask, word_id, word_mask, bbox = dataset[0]
        print(f"img:       {img.shape}, dtype={img.dtype}")
        print(f"img_mask:  {img_mask.shape}, dtype={img_mask.dtype}")
        print(f"word_id:   {word_id.shape}")
        print(f"word_mask: {word_mask.shape}")
        print(f"bbox:      {bbox}")
        print("\n✅ Dataset test passed!")
    except FileNotFoundError:
        print("⚠️ Data files not found (normal on local machine)")
        print("✅ Dataset code OK, test on Kaggle")
