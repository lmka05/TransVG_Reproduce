# ==============================================================================
# config.py — Cấu hình cho TransVG Reimplementation
# ==============================================================================
# Tất cả hyperparameters tập trung ở đây.
# Khi muốn thay đổi bất kỳ tham số nào → sửa file này.
# ==============================================================================

class Config:
    # ==========================================================================
    # 1. ĐƯỜNG DẪN DỮ LIỆU (Kaggle)
    # ==========================================================================
    img_dir = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
    ann_file = "/kaggle/input/datasets/minhkhoai/seqtr-annotations-weights/annotations/refcoco-unc/instances.json"
    detr_model = "/kaggle/input/datasets/minhkhoai/detr-pretrained/detr-r50-e632da11.pth"
    # ==========================================================================
    # 2. KÍCH THƯỚC ĐẦU VÀO
    # ==========================================================================
    imsize = 640             # Ảnh resize + pad về 640×640
    max_query_len = 15       # Số từ tối đa trong câu (không tính [CLS], [SEP])

    # ==========================================================================
    # 3. VISUAL BACKBONE (ResNet-50 + DETR Encoder)
    # ==========================================================================
    backbone = "resnet50"
    dilation = False         # False → stride=32, feature map 20×20
    hidden_dim = 256         # Dimension sau Conv1x1 (2048 → 256)
    nheads = 8               # Số attention heads trong DETR Encoder
    dim_feedforward = 2048   # FFN hidden dim trong DETR Encoder
    dropout = 0.1
    pre_norm = False         # Post-norm (mặc định)
    detr_enc_num = 6         # Số encoder layers trong DETR
    position_embedding = "sine"  # Loại positional encoding ("sine" = sin/cos, không cần train)

    # ==========================================================================
    # 4. LANGUAGE ENCODER (BERT)
    # ==========================================================================
    bert_model = "bert-base-uncased"
    bert_enc_num = 12        # Dùng output layer thứ 12

    # ==========================================================================
    # 5. VISION-LANGUAGE TRANSFORMER
    # ==========================================================================
    vl_hidden_dim = 256
    vl_nheads = 8
    vl_enc_layers = 6
    vl_dim_feedforward = 2048
    vl_dropout = 0.1

    # ==========================================================================
    # 6. TRAINING
    # ==========================================================================
    optimizer = "adamw"      # Optimizer (chỉ dùng AdamW)
    # [CŨ] LR theo paper (batch=64): lr=1e-4, lr_bert/cnn/tra=1e-5
    # [MỚI] Scale xuống theo Linear Scaling Rule: lr_new = lr_paper × (batch/batch_paper)
    #        = lr_paper × (8/64) = lr_paper × 0.125
    lr          = 1.25e-5    # LR cho VL Transformer + MLP  (1e-4  × 8/64)
    lr_bert     = 1.25e-6    # LR cho BERT                  (1e-5  × 8/64)
    lr_visu_cnn = 1.25e-6    # LR cho ResNet backbone        (1e-5  × 8/64)
    lr_visu_tra = 1.25e-6    # LR cho DETR Encoder           (1e-5  × 8/64)
    weight_decay = 1e-4
    batch_size = 8
    epochs = 90
    lr_scheduler = "step"    # "step", "cosine" 
    lr_drop = 60             # Epoch giảm lr (cho step scheduler)
    clip_max_norm = 0.15     # Gradient clipping

    # ==========================================================================
    # 7. LOGGING & CHECKPOINT
    # ==========================================================================
    log_interval = 80        # In log mỗi N batches
    output_dir = "/kaggle/working/transvg_outputs"
    resume = "  "              # Đường dẫn checkpoint để resume training

    # ==========================================================================
    # 8. MISC
    # ==========================================================================
    seed = 13
    num_workers = 2
    device = "cuda"
