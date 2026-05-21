# ==============================================================================
# box_utils.py — Bounding Box Utilities
# ==============================================================================
# Tất cả hàm liên quan đến bounding box:
#   - Chuyển đổi format: xywh ↔ xyxy
#   - Tính IoU (Intersection over Union)
#   - Tính GIoU (Generalized IoU — dùng trong loss function)
#
# TransVG dùng 2 format bbox:
#   - [x_c, y_c, w, h]: Tâm + kích thước (model output, normalized [0,1])
#   - [x1, y1, x2, y2]: Góc trên-trái + góc dưới-phải (dùng khi tính IoU)
# ==============================================================================

import torch
from torchvision.ops.boxes import box_area


def xywh2xyxy(x):
    """
    Chuyển bbox từ [x_center, y_center, width, height] → [x1, y1, x2, y2].

    Args:
        x (Tensor): [..., 4] — bbox dạng center format

    Returns:
        Tensor: [..., 4] — bbox dạng corner format

    Ví dụ:
        [0.5, 0.5, 0.4, 0.6] → [0.3, 0.2, 0.7, 0.8]
        Tâm (0.5, 0.5), rộng 0.4, cao 0.6
        → x1 = 0.5 - 0.2 = 0.3, y1 = 0.5 - 0.3 = 0.2
        → x2 = 0.5 + 0.2 = 0.7, y2 = 0.5 + 0.3 = 0.8
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy2xywh(x):
    """
    Chuyển bbox từ [x1, y1, x2, y2] → [x_center, y_center, width, height].

    Args:
        x (Tensor): [..., 4] — bbox dạng corner format

    Returns:
        Tensor: [..., 4] — bbox dạng center format

    Ví dụ:
        [0.3, 0.2, 0.7, 0.8] → [0.5, 0.5, 0.4, 0.6]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def bbox_iou(box1, box2):
    """
    Tính IoU giữa 2 tập bounding boxes.

    IoU (Intersection over Union) = Diện tích giao / Diện tích hợp
    Được dùng để đánh giá model: IoU ≥ 0.5 → dự đoán đúng.

    Args:
        box1 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        box2 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        (N phải bằng nhau — so sánh 1-1)

    Returns:
        Tensor: [N] — giá trị IoU cho từng cặp box

    Ví dụ:
        box1 = [[0, 0, 100, 100]]
        box2 = [[50, 50, 150, 150]]
        intersection = 50 * 50 = 2500
        union = 10000 + 10000 - 2500 = 17500
        IoU = 2500 / 17500 ≈ 0.143
    """
    # Tọa độ của từng box
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Tọa độ vùng giao (intersection)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # Diện tích giao (clamp để tránh giá trị âm khi không giao nhau)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    # Diện tích của từng box
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # IoU = intersection / union
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def box_iou_pairwise(boxes1, boxes2):
    """
    Tính IoU pairwise giữa 2 tập boxes (N×M matrix).

    Args:
        boxes1 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        boxes2 (Tensor): [M, 4] — dạng [x1, y1, x2, y2]

    Returns:
        iou (Tensor): [N, M] — ma trận IoU
        union (Tensor): [N, M] — diện tích union
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Tìm giao giữa mọi cặp (N, M)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2] — left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2] — right-bottom

    wh = (rb - lt).clamp(min=0)       # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter  # [N, M]

    iou = inter / (union + 1e-16)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Tính Generalized IoU (GIoU) giữa 2 tập boxes.

    GIoU = IoU - (Area_C - Union) / Area_C
    Trong đó Area_C là diện tích hình chữ nhật bao quanh (enclosing box).

    GIoU ∈ [-1, 1]:
        - GIoU = 1: trùng hoàn toàn
        - GIoU = 0: giao nhau 1 phần
        - GIoU < 0: không giao nhau (xa nhau → GIoU → -1)

    Ưu điểm so với IoU thông thường:
        - IoU = 0 khi 2 box không giao → gradient = 0 → model không học được
        - GIoU vẫn có giá trị < 0 khi không giao → gradient vẫn chảy

    Args:
        boxes1 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]
        boxes2 (Tensor): [N, 4] — dạng [x1, y1, x2, y2]

    Returns:
        Tensor: [N, N] — ma trận GIoU

    Dùng trong loss:
        loss_giou = 1 - diag(GIoU(pred, target))
    """
    # Kiểm tra box hợp lệ (x2 >= x1, y2 >= y1)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1: x2 phải >= x1, y2 phải >= y1"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2: x2 phải >= x1, y2 phải >= y1"

    # Tính IoU và union
    iou, union = box_iou_pairwise(boxes1, boxes2)

    # Tính enclosing box (bao quanh cả 2 box)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # [N, M] — diện tích enclosing box

    # GIoU = IoU - (C - Union) / C
    giou = iou - (area_c - union) / (area_c + 1e-16)
    return giou


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=== Test box_utils ===")

    # Test xywh ↔ xyxy
    xywh = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
    xyxy = xywh2xyxy(xywh)
    print(f"xywh2xyxy: {xywh} → {xyxy}")
    # Expected: [0.3, 0.2, 0.7, 0.8]

    back = xyxy2xywh(xyxy)
    print(f"xyxy2xywh: {xyxy} → {back}")
    # Expected: [0.5, 0.5, 0.4, 0.6]

    # Test IoU
    box1 = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    box2 = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
    iou = bbox_iou(box1, box2)
    print(f"IoU: {iou.item():.4f}")  # ~0.1429

    # Test GIoU
    giou = generalized_box_iou(box1, box2)
    print(f"GIoU: {giou.item():.4f}")

    # Test trường hợp không giao nhau
    box3 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box4 = torch.tensor([[90.0, 90.0, 100.0, 100.0]])
    iou_no_overlap = bbox_iou(box3, box4)
    giou_no_overlap = generalized_box_iou(box3, box4)
    print(f"No overlap — IoU: {iou_no_overlap.item():.4f}, GIoU: {giou_no_overlap.item():.4f}")

    print("\n✅ box_utils test passed!")
