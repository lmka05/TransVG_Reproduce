# ==============================================================================
# test.py — Test TransVG trên val/testA/testB
# ==============================================================================
# Chạy: python test.py --checkpoint /path/to/best.pth
# Style giống SeqTR.
# ==============================================================================

import os
import sys
import argparse

import torch

from config import Config
from models import build_model
from datasets import build_val_loader
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='TransVG — Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--splits', nargs='+', default=['val', 'testA', 'testB'],
                        help='Các split cần đánh giá (mặc định: val testA testB)')
    args = parser.parse_args()

    config = Config
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Splits: {args.splits}")

    # 1. Build model
    print("\n--- Building model ---")
    model = build_model(config)
    model.to(device)

    # 2. Load checkpoint
    print("\n--- Loading checkpoint ---")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    best_acc = ckpt.get('best_accuracy', '?')
    print(f"Loaded checkpoint: epoch={epoch}, best_acc={best_acc}")

    # 3. Evaluate từng split
    results = {}
    for split in args.splits:
        print(f"\n--- Evaluating on [{split}] ---")
        try:
            loader = build_val_loader(config, split=split)
            acc, avg_iou = evaluate(model, loader, device, desc=split)
            results[split] = {'accuracy': acc, 'avg_iou': avg_iou}
        except KeyError:
            print(f"  ⚠️ Split '{split}' không tồn tại trong file annotations.")

    # 4. Tổng kết (giống SeqTR)
    print("\n" + "=" * 60)
    print("TỔNG KẾT KẾT QUẢ")
    print("=" * 60)
    for split, res in results.items():
        print(f"  {split:8s}: Acc@IoU>=0.5 = {res['accuracy']:.2f}% | "
              f"Avg IoU = {res['avg_iou']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
