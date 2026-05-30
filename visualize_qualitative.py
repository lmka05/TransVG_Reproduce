# ==============================================================================
# visualize_qualitative.py — Qualitative Results cho TransVG
# ==============================================================================
# Vẽ bounding box Ground Truth (đỏ) và Prediction (xanh) lên ảnh gốc.
# Hiển thị IoU, referring expression cho từng ảnh.
# Ghép N ảnh thành 1 figure để dùng trong báo cáo/paper.
#
# Cách chạy:
#   python visualize_qualitative.py \
#       --checkpoint transvg-checkpoint/best.pth \
#       --split val \
#       --num_images 4 \
#       --output qualitative_results.png
#
# Xem thêm tùy chọn: python visualize_qualitative.py --help
# ==============================================================================

import os
import sys
import json
import random
import argparse

import numpy as np
from PIL import Image

import torch

import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI (tương thích Kaggle)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import Config
from models import build_model
from utils.box_utils import xywh2xyxy
from utils.image_transforms import (
    resize_image_keep_ratio,
    pad_image_to_square,
    normalize_imagenet,
    image_to_tensor,
    create_image_mask,
)
from utils.text_transforms import TextTransform


# ==============================================================================
# 1. INFERENCE CHO 1 ẢNH
# ==============================================================================

def predict_single(model, img_tensor, img_mask, word_ids, word_mask, device):
    """
    Chạy inference cho 1 sample.

    Args:
        model: TransVG model (đã eval)
        img_tensor: [1, 3, 640, 640]
        img_mask:   [1, 640, 640]
        word_ids:   [1, 17]
        word_mask:  [1, 17]
        device: cuda/cpu

    Returns:
        pred_box: Tensor [4] — normalized (cx, cy, w, h) ∈ [0,1]
    """
    img_tensor = img_tensor.to(device)
    img_mask = img_mask.to(device)
    word_ids = word_ids.to(device)
    word_mask = word_mask.to(device)

    with torch.no_grad():
        pred_box = model(img_tensor, img_mask, word_ids, word_mask)  # [1, 4]

    return pred_box[0].cpu()  # [4]


# ==============================================================================
# 2. TIỀN XỬ LÝ 1 SAMPLE (không qua Dataset class)
# ==============================================================================

def preprocess_sample(ann, img_dir, config, text_transform):
    """
    Tiền xử lý 1 annotation thành input cho model + thông tin để vẽ.

    Args:
        ann (dict): 1 annotation từ instances.json
        img_dir (str): Thư mục ảnh COCO
        config: Config object
        text_transform: TextTransform instance

    Returns:
        dict chứa:
            - img_tensor:  [1, 3, 640, 640]
            - img_mask:    [1, 640, 640]
            - word_ids:    [1, 17]
            - word_mask:   [1, 17]
            - gt_bbox_orig: [4] — GT xyxy trên ảnh gốc (pixel)
            - expression:  str — câu mô tả
            - pil_img:     PIL.Image — ảnh gốc (để vẽ)
            - scale:       float — tỉ lệ resize
    """
    imsize = config.imsize  # 640

    # --- Load ảnh gốc ---
    img_path = os.path.join(
        img_dir,
        "COCO_train2014_%012d.jpg" % ann['image_id']
    )
    pil_img = Image.open(img_path).convert('RGB')
    img = np.array(pil_img)

    # --- Resize + Pad + Normalize ---
    img_resized, scale = resize_image_keep_ratio(img, imsize)
    resized_h, resized_w = img_resized.shape[:2]

    img_padded = pad_image_to_square(img_resized, imsize)
    img_norm = normalize_imagenet(img_padded)
    img_tensor = image_to_tensor(img_norm).unsqueeze(0)  # [1, 3, 640, 640]

    # --- Image mask ---
    img_mask = create_image_mask(resized_h, resized_w, imsize).unsqueeze(0)  # [1, 640, 640]

    # --- Text ---
    expression = ann['expressions'][0]  # Luôn lấy câu đầu (consistent)
    input_ids, attention_mask = text_transform(expression)
    word_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)       # [1, 17]
    word_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)  # [1, 17]

    # --- Ground Truth bbox (COCO → xyxy trên ảnh gốc) ---
    x, y, w, h = ann['bbox']
    gt_bbox_orig = [x, y, x + w, y + h]  # [x1, y1, x2, y2] pixel gốc

    return {
        'img_tensor': img_tensor,
        'img_mask': img_mask,
        'word_ids': word_ids,
        'word_mask': word_mask,
        'gt_bbox_orig': gt_bbox_orig,
        'expression': expression,
        'pil_img': pil_img,
        'scale': scale,
    }


# ==============================================================================
# 3. CHUYỂN PREDICTION VỀ TỌA ĐỘ ẢNH GỐC
# ==============================================================================

def pred_to_orig_coords(pred_box_norm, scale, imsize=640):
    """
    Chuyển pred_box từ normalized (cx,cy,w,h) → xyxy pixel trên ảnh gốc.

    Pipeline:
        normalized cxcywh → ×640 → xyxy trên ảnh 640×640 → ÷scale → xyxy ảnh gốc

    Args:
        pred_box_norm: Tensor [4] — normalized (cx, cy, w, h)
        scale: float — tỉ lệ resize (original → 640)
        imsize: int — kích thước ảnh pad (640)

    Returns:
        list [x1, y1, x2, y2] — pixel trên ảnh gốc
    """
    # Normalized → pixel trên ảnh 640×640
    pred_xyxy_640 = xywh2xyxy(pred_box_norm.unsqueeze(0))[0] * imsize

    # Pixel 640×640 → pixel ảnh gốc
    pred_xyxy_orig = pred_xyxy_640 / scale

    return pred_xyxy_orig.tolist()  # [x1, y1, x2, y2]


# ==============================================================================
# 4. TÍNH IoU GIỮA 2 BOX (xyxy format)
# ==============================================================================

def compute_iou(box1, box2):
    """
    Tính IoU giữa 2 box ở dạng [x1, y1, x2, y2].

    Args:
        box1, box2: list/tuple [x1, y1, x2, y2]

    Returns:
        float: IoU ∈ [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# ==============================================================================
# 5. VẼ 1 ẢNH VỚI BBOX
# ==============================================================================

def draw_single_result(ax, pil_img, gt_bbox, pred_bbox, expression, iou, label_idx=None):
    """
    Vẽ 1 ảnh lên 1 subplot với GT bbox (đỏ) và Pred bbox (xanh).

    Args:
        ax: matplotlib Axes
        pil_img: PIL.Image — ảnh gốc
        gt_bbox: [x1, y1, x2, y2] pixel gốc
        pred_bbox: [x1, y1, x2, y2] pixel gốc
        expression: str — referring expression
        iou: float — IoU score
        label_idx: int hoặc None — index cho nhãn (a), (b), (c)...
    """
    ax.imshow(pil_img)

    # --- Vẽ GT bbox (đỏ) ---
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    gt_rect = patches.Rectangle(
        (gt_x1, gt_y1), gt_x2 - gt_x1, gt_y2 - gt_y1,
        linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-'
    )
    ax.add_patch(gt_rect)

    # --- Vẽ Pred bbox (xanh) ---
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
    pred_rect = patches.Rectangle(
        (pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1,
        linewidth=2.5, edgecolor='blue', facecolor='none', linestyle='-'
    )
    ax.add_patch(pred_rect)

    # --- Hiển thị IoU (góc trên trái) ---
    ax.text(
        5, 5, f"IoU: {iou:.2f}",
        fontsize=12, fontweight='bold',
        color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75)
    )

    # --- Hiển thị expression (tiêu đề) ---
    if label_idx is not None:
        label_char = chr(ord('a') + label_idx)
        title = f"({label_char}) {expression}"
    else:
        title = expression
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    # --- Ẩn trục ---
    ax.axis('off')


# ==============================================================================
# 6. MAIN — GHÉP FIGURE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TransVG — Qualitative Results Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python visualize_qualitative.py --checkpoint transvg-checkpoint/best.pth --split val --num_images 4
  python visualize_qualitative.py --checkpoint transvg-checkpoint/best.pth --split testA --num_images 8 --cols 4
  python visualize_qualitative.py --checkpoint transvg-checkpoint/best.pth --split testB --num_images 6 --cols 3 --seed 42
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'testA', 'testB'],
                        help='Split dataset (mặc định: val)')
    parser.add_argument('--num_images', type=int, default=4,
                        help='Số lượng ảnh visualize (mặc định: 4)')
    parser.add_argument('--cols', type=int, default=None,
                        help='Số cột trong figure (mặc định: bằng num_images nếu ≤4, ngược lại 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Đường dẫn lưu ảnh output (mặc định: qualitative_{split}.png)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (để reproducible)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='DPI cho ảnh output (mặc định: 200)')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Đường dẫn thư mục ảnh COCO (mặc định: dùng config)')
    parser.add_argument('--ann_file', type=str, default=None,
                        help='Đường dẫn file annotation JSON (mặc định: dùng config)')

    args = parser.parse_args()

    # --- Config ---
    config = Config
    img_dir = args.img_dir if args.img_dir else config.img_dir
    ann_file = args.ann_file if args.ann_file else config.ann_file
    output_path = args.output if args.output else f"qualitative_{args.split}.png"
    cols = args.cols if args.cols else min(args.num_images, 4)

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Num images: {args.num_images}")
    print(f"Image dir: {img_dir}")
    print(f"Annotation file: {ann_file}")

    # ==========================================================================
    # STEP 1: Load model + checkpoint
    # ==========================================================================
    print("\n[1/5] Loading model...")
    model = build_model(config)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    best_acc = ckpt.get('best_accuracy', '?')
    print(f"  Loaded: epoch={epoch}, best_acc={best_acc}")

    # ==========================================================================
    # STEP 2: Load annotations + chọn random N samples
    # ==========================================================================
    print(f"\n[2/5] Loading annotations [{args.split}]...")
    anns_all = json.load(open(ann_file, 'r'))
    anns = anns_all[args.split]
    print(f"  Total samples in [{args.split}]: {len(anns)}")

    # Random chọn N samples
    num_images = min(args.num_images, len(anns))
    selected_indices = random.sample(range(len(anns)), num_images)
    print(f"  Selected indices: {selected_indices}")

    # ==========================================================================
    # STEP 3: Text transform (BERT tokenizer)
    # ==========================================================================
    print("\n[3/5] Initializing BERT tokenizer...")
    text_transform = TextTransform(
        bert_model=config.bert_model,
        max_query_len=config.max_query_len
    )

    # ==========================================================================
    # STEP 4: Inference + thu thập kết quả
    # ==========================================================================
    print(f"\n[4/5] Running inference on {num_images} images...")
    results = []

    for i, idx in enumerate(selected_indices):
        ann = anns[idx]
        print(f"  [{i+1}/{num_images}] image_id={ann['image_id']}, "
              f"expr=\"{ann['expressions'][0][:50]}...\"")

        # Tiền xử lý
        sample = preprocess_sample(ann, img_dir, config, text_transform)

        # Inference
        pred_box_norm = predict_single(
            model,
            sample['img_tensor'],
            sample['img_mask'],
            sample['word_ids'],
            sample['word_mask'],
            device
        )

        # Chuyển prediction về toạ độ ảnh gốc
        pred_bbox_orig = pred_to_orig_coords(
            pred_box_norm, sample['scale'], config.imsize
        )

        # GT bbox đã sẵn ở toạ độ ảnh gốc
        gt_bbox_orig = sample['gt_bbox_orig']

        # Tính IoU
        iou = compute_iou(gt_bbox_orig, pred_bbox_orig)

        results.append({
            'pil_img': sample['pil_img'],
            'gt_bbox': gt_bbox_orig,
            'pred_bbox': pred_bbox_orig,
            'expression': sample['expression'],
            'iou': iou,
        })

        print(f"    → IoU = {iou:.4f} | GT={[f'{v:.1f}' for v in gt_bbox_orig]} | "
              f"Pred={[f'{v:.1f}' for v in pred_bbox_orig]}")

    # ==========================================================================
    # STEP 5: Vẽ figure
    # ==========================================================================
    print(f"\n[5/5] Drawing figure...")
    rows = (num_images + cols - 1) // cols  # Ceiling division
    fig_width = 5.5 * cols
    fig_height = 5 * rows + 0.8  # Thêm chỗ cho legend

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Đảm bảo axes luôn là 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # Vẽ từng ảnh
    for i, result in enumerate(results):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        draw_single_result(
            ax,
            result['pil_img'],
            result['gt_bbox'],
            result['pred_bbox'],
            result['expression'],
            result['iou'],
            label_idx=i
        )

    # Ẩn subplot thừa (nếu num_images không chia hết cho cols)
    for i in range(num_images, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')

    # --- Legend chung ở dưới cùng ---
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='blue', linewidth=2.5, label='Prediction'),
        patches.Patch(facecolor='none', edgecolor='red', linewidth=2.5, label='Ground Truth'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=2,
        fontsize=13,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=0.8,
        handlelength=2.0,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # Chừa chỗ cho legend
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n{'='*60}")
    print(f"✅ Saved: {output_path}")
    print(f"   Split: {args.split}")
    print(f"   Images: {num_images}")
    print(f"   Avg IoU: {sum(r['iou'] for r in results) / len(results):.4f}")
    print(f"   Acc@0.5: {sum(1 for r in results if r['iou'] >= 0.5)}/{len(results)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
