# ==============================================================================
# evaluate.py — Loss function + Đánh giá model
# ==============================================================================
# 2 phần:
#   1. trans_vg_loss()    — L1 + GIoU loss (dùng khi training)
#   2. evaluate()         — Tính Acc@0.5 + mIoU (dùng khi val/test)
#
# Style giống SeqTR: vòng for đơn giản, print trực tiếp.
# ==============================================================================

import torch
import torch.nn.functional as F

from utils.box_utils import xywh2xyxy, generalized_box_iou


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

def trans_vg_loss(pred_boxes, target_boxes):
    """
    Loss = L1 + GIoU.

    Args:
        pred_boxes:   [B, 4] — model output, normalized (x_c, y_c, w, h)
        target_boxes: [B, 4] — ground truth, normalized (x_c, y_c, w, h)

    Returns:
        dict: {'loss_bbox': L1_loss, 'loss_giou': GIoU_loss}
    """
    batch_size = pred_boxes.shape[0]

    # L1 Loss
    loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='none')
    loss_bbox = loss_bbox.sum() / batch_size

    # GIoU Loss
    pred_xyxy = xywh2xyxy(pred_boxes)
    target_xyxy = xywh2xyxy(target_boxes)
    giou_matrix = generalized_box_iou(pred_xyxy, target_xyxy)
    loss_giou = 1 - torch.diag(giou_matrix)
    loss_giou = loss_giou.sum() / batch_size

    return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}


# ==============================================================================
# IoU TÍNH TOÁN (giống SeqTR)
# ==============================================================================

def compute_iou_batch(pred, gt):
    """
    Tính IoU cho 1 batch. Code giống y SeqTR.

    Args:
        pred (Tensor): [B, 4] — [x1, y1, x2, y2]
        gt (Tensor):   [B, 4] — [x1, y1, x2, y2]

    Returns:
        iou (Tensor): [B]
    """
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


# ==============================================================================
# EVALUATE (giống SeqTR)
# ==============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating"):
    """
    Đánh giá model trên 1 split. Style giống SeqTR.

    Args:
        model: TransVG model
        dataloader: DataLoader
        device: 'cuda' hoặc 'cpu'
        desc: Tên split (dùng in log)

    Returns:
        accuracy (float): Acc@IoU>=0.5 (%)
        avg_iou (float): IoU trung bình (%)
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_iou = 0.0

    for batch_idx, (img_data, text_data, target) in enumerate(dataloader):
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        # Forward
        # [CŨ] pred_boxes = model(img_data, text_data)
        # [MỚI] Tách NestedTensor → 4 tensor thuần (tương thích DataParallel)
        pred_boxes = model(img_data.tensors, img_data.mask,
                           text_data.tensors, text_data.mask)  # [B, 4] normalized xywh

        # Chuyển sang xyxy để tính IoU
        pred_xyxy = xywh2xyxy(pred_boxes)
        target_xyxy = xywh2xyxy(target)

        # Tính IoU
        iou = compute_iou_batch(pred_xyxy, target_xyxy)  # [B]

        # Đếm
        correct = (iou >= 0.5).sum().item()
        total_correct += correct
        total_samples += pred_boxes.shape[0]
        total_iou += iou.sum().item()

    accuracy = total_correct / total_samples * 100
    avg_iou = total_iou / total_samples * 100

    print(f"  [{desc}] Acc@IoU>=0.5: {accuracy:.2f}% | Avg IoU: {avg_iou:.2f}% | Samples: {total_samples}")

    return accuracy, avg_iou
