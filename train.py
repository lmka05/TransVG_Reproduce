# ==============================================================================
# train.py — Training Script cho TransVG
# ==============================================================================
# Entry point chính — chạy: python train.py
#
# Style giống SeqTR:
#   - Vòng for đơn giản (không dùng MetricLogger)
#   - Log format: Epoch 13 | Batch 80/2650 | Loss: 4.4309 | LR: 0.000063 |
#   - Checkpoint save/resume
#
# Khác SeqTR:
#   - 4 param groups (lr khác nhau cho ResNet/DETR/BERT/rest)
#   - Loss = L1 + GIoU (thay vì CrossEntropy)
#   - Không có EMA (đơn giản hơn)
# ==============================================================================

import os
import torch.nn as nn
import sys
import math
import time
import json
import random
import gc

import numpy as np
import torch

from config import Config
from models import build_model
from datasets import build_train_loader, build_val_loader
from evaluate import trans_vg_loss, evaluate


# ==============================================================================
# PHẦN 1: TIỆN ÍCH (giống SeqTR)
# ==============================================================================

def set_seed(seed):
    """Đặt random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, best_accuracy, config):
    """Lưu checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    # [MỚI] Lấy model gốc bên trong DataParallel (nếu có)
    # DataParallel wrap thêm 1 lớp .module → state_dict keys sẽ có prefix 'module.'
    # → load lại trên 1 GPU sẽ lỗi. Luôn save model gốc.
    # [CŨ] 'model_state_dict': model.state_dict(),
    raw_model = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy,
    }

    # Luôn lưu latest (để resume)
    latest_path = os.path.join(config.output_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)

    # Lưu best nếu accuracy cao nhất
    if accuracy >= best_accuracy:
        best_path = os.path.join(config.output_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model saved! Acc: {accuracy:.2f}%")


# ==============================================================================
# PHẦN 2: TRAIN 1 EPOCH (giống SeqTR)
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    """
    Train model qua 1 epoch.

    Log format: Epoch 13 | Batch 80/2650 | Loss: 4.4309 | LR: 0.000063 |

    Returns:
        avg_loss (float): Loss trung bình của epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (img_data, text_data, target) in enumerate(dataloader):
        # 1. Move to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # 2. Forward
        # [CŨ] pred_box = model(img_data, text_data)
        # [MỚI] Tách NestedTensor → 4 tensor thuần (để DataParallel chia batch được)
        pred_box = model(img_data.tensors, img_data.mask,
                         text_data.tensors, text_data.mask)  # [B, 4]

        # 3. Loss
        losses = trans_vg_loss(pred_box, target)
        loss = losses['loss_bbox'] + losses['loss_giou']

        # [MỚI] DataParallel trả loss từ mỗi GPU → cần mean
        if loss.dim() > 0:
            loss = loss.mean()

        loss_value = loss.item()

        # 4. Check NaN
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # 5. Backward
        optimizer.zero_grad()
        loss.backward()
        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
        optimizer.step()

        # 6. Tracking
        total_loss += loss_value
        num_batches += 1

        # [FIX] Dọn dẹp RAM ngay trong loop để không rò rỉ rác giữa các batch
        del img_data, text_data, target, pred_box, losses, loss
        if batch_idx % 100 == 0:
            gc.collect()
            
        # 7. Log (giống SeqTR)
        if (batch_idx + 1) % config.log_interval == 0:
            avg = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    return avg_loss


# ==============================================================================
# PHẦN 3: MAIN (giống SeqTR)
# ==============================================================================

def main():
    config = Config

    # 0. Seed
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Build model
    print("\n" + "=" * 60)
    print("STEP 1: Building model")
    print("=" * 60)
    model = build_model(config)
    model.to(device)

    # [MỚI] Load DETR pretrain nếu có
    if os.path.exists(config.detr_model):
        print(f"\n--- Loading DETR pretrain from {config.detr_model} ---")
        checkpoint = torch.load(config.detr_model, map_location='cpu', weights_only=False)
        # Load đè vào nhánh visual của model (model.visumodel chính là VisualEncoder)
        # strict=False vì cấu trúc có thể có vài key thừa/thiếu nhỏ xíu
        missing_keys, unexpected_keys = model.visumodel.load_state_dict(checkpoint['model'], strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print("=> Đã tải thành công tệp DETR pretrain cho ResNet-50 và DETR Encoder!")
    else:
        print(f"\n⚠️ CẢNH BÁO: Không tìm thấy file DETR pretrain tại: {config.detr_model}")
        print("=> Model sẽ train Visual nhánh từ đầu (hoặc ImageNet)!")
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    # [MỚI] Multi-GPU: wrap model bằng DataParallel (giống SeqTR)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n🚀 Using {num_gpus} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    else:
        print(f"\nUsing 1 GPU")

    # 2. Optimizer — 4 param groups
    print("\n" + "=" * 60)
    print("STEP 2: Setting up optimizer")
    print("=" * 60)

    # ResNet backbone → fine-tune nhẹ
    visu_cnn_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" in n and p.requires_grad]
    # DETR Encoder → fine-tune nhẹ
    visu_tra_param = [p for n, p in model.named_parameters()
                      if "visumodel" in n and "backbone" not in n and p.requires_grad]
    # BERT → fine-tune nhẹ
    text_tra_param = [p for n, p in model.named_parameters()
                      if "textmodel" in n and p.requires_grad]
    # Còn lại (VL Transformer + projections + [REG] + MLP) → train mạnh
    rest_param = [p for n, p in model.named_parameters()
                  if "visumodel" not in n and "textmodel" not in n and p.requires_grad]

    param_list = [
        {"params": rest_param,       "lr": config.lr},
        {"params": visu_cnn_param,   "lr": config.lr_visu_cnn},
        {"params": visu_tra_param,   "lr": config.lr_visu_tra},
        {"params": text_tra_param,   "lr": config.lr_bert},
    ]

    print(f"  rest (VL Trans+MLP):  {sum(p.numel() for p in rest_param):>10,} params, lr={config.lr}")
    print(f"  visu_cnn (ResNet):    {sum(p.numel() for p in visu_cnn_param):>10,} params, lr={config.lr_visu_cnn}")
    print(f"  visu_tra (DETR Enc):  {sum(p.numel() for p in visu_tra_param):>10,} params, lr={config.lr_visu_tra}")
    print(f"  text_tra (BERT):      {sum(p.numel() for p in text_tra_param):>10,} params, lr={config.lr_bert}")

    optimizer = torch.optim.AdamW(param_list, lr=config.lr, weight_decay=config.weight_decay)

    # 3. LR Scheduler
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    elif config.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / config.epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    # 4. Build dataloaders
    print("\n" + "=" * 60)
    print("STEP 3: Creating datasets")
    print("=" * 60)
    train_loader = build_train_loader(config)
    val_loader = build_val_loader(config, split='val')

    # 5. Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    latest_ckpt = os.path.join(config.output_dir, 'latest.pth')

    if config.resume and os.path.isfile(config.resume):
        latest_ckpt = config.resume

    if os.path.exists(latest_ckpt):
        print(f"\nResuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        # [MỚI] Load vào model gốc (bên trong DataParallel nếu có)
        # [CŨ] model.load_state_dict(ckpt['model_state_dict'])
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_accuracy:.2f}%")

    # 6. Training loop
    print("\n" + "=" * 60)
    print("STEP 4: Start training!")
    print("=" * 60)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)

        # Evaluate
        print(f"\n  --- Evaluating epoch {epoch+1} ---")
        val_acc, val_iou = evaluate(model, val_loader, device, desc="val")

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_accuracy, config)
        best_accuracy = max(best_accuracy, val_acc)

        # Step scheduler
        scheduler.step()

        # Giải phóng GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Epoch summary (giống SeqTR)
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_accuracy:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.0f}s")
        print(f"{'='*60}\n")

    print(f"\n🎉 Training finished! Best accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved at: {config.output_dir}")


if __name__ == "__main__":
    main()
