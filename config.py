# coding=utf-8
import torch

# ==================================================================================
# 参数配置
# ==================================================================================

# 硬件设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ViT-MAE 模型参数
model_params = {
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "intermediate_size": 384 * 4,
    # Decoder 参数
    "decoder_hidden_size": 256,
    "decoder_num_hidden_layers": 4,
    "decoder_num_attention_heads": 4,
    "decoder_intermediate_size": 256 * 4,
    "norm_pix_loss": True,
    "mask_ratio": 0.65,
}

# 训练参数
training_params = {
    "data_dir": "",
    "output_dir": "./output",
    "num_epochs": 200,
    "batch_size": 128,
    "learning_rate": 1.5e-4 * 128 / 256,
    "weight_decay": 0.05,
    "num_workers": 0,  # Windows下建议设为0
    "pin_memory": True,
    "base_processor": "google/vit-base-patch16-224-in21k",
}

# 损失函数参数
dct_loss_params = {
    "lambda_dct": 0.5,
    "min_weight": 0.1,
    "power": 2.0
}

loss_weights = {
    "perceptual": 0.05,
    "gradient": 1.0
}

# 保存策略
saving_params = {
    "save_best_only": False,
    "num_checkpoints_to_keep": 2
}