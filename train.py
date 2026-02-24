# coding=utf-8
import os
import shutil
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTMAEConfig, ViTMAEForPreTraining, AutoImageProcessor

# --- 新增：混合精度训练所需模块 ---
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from config import (
    model_params, training_params, dct_loss_params,
    loss_weights, saving_params, SEED, DEVICE
)
from dataset import InMemoryImageFolder
from losses import DCT2D, PerceptualLoss, gradient_loss
from utils import set_seed, generate_detailed_visualizations, plot_loss_curves


def main():
    # 0. 环境设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 显存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_seed(SEED)
    os.makedirs(training_params["output_dir"], exist_ok=True)
    device = DEVICE

    print(f"Initializing on {device}...")

    # 1. 模型与处理器
    image_processor = AutoImageProcessor.from_pretrained(training_params["base_processor"], use_fast=True)
    config = ViTMAEConfig(**model_params)
    model = ViTMAEForPreTraining(config)
    model.to(device)

    # 2. 数据准备
    transform = transforms.Compose([
        transforms.RandomResizedCrop(model_params["image_size"], scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    dataset = InMemoryImageFolder(root=training_params["data_dir"], transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=training_params["num_workers"],
        pin_memory=training_params["pin_memory"]
    )

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
        betas=(0.9, 0.95)
    )

    # --- 新增：初始化 GradScaler ---
    scaler = GradScaler()

    # 4. 损失组件初始化
    patch_size = model_params["patch_size"]
    dct_layer = DCT2D(size=patch_size).to(device)
    perceptual_loss_func = PerceptualLoss(device)

    # 预计算 DCT 权重
    i_c = torch.arange(patch_size, device=device, dtype=torch.float32)
    i_g, j_g = torch.meshgrid(i_c, i_c, indexing='ij')
    dist = torch.sqrt(i_g ** 2 + j_g ** 2)
    norm_dist = dist / dist.max()
    dct_w = dct_loss_params["min_weight"] + (1 - dct_loss_params["min_weight"]) * (
            norm_dist ** dct_loss_params["power"])
    dct_w = dct_w.detach()

    # 5. 记录初始化
    history = {
        'epoch': [], 'total_loss': [], 'pixel_loss': [],
        'dct_loss': [], 'perceptual_loss': [], 'grad_loss': []
    }
    loss_log_path = os.path.join(training_params["output_dir"], "log.xlsx")
    loss_df = pd.DataFrame(
        columns=['Epoch', 'Total Loss', 'Pixel Loss', 'Weighted DCT Loss', 'Perceptual Loss', 'Gradient Loss'])
    best_loss = float('inf')

    print(f"Start Mixed Precision Training...")

    # 6. 训练循环
    for epoch in range(training_params["num_epochs"]):
        model.train()
        avg = {'total': 0, 'pixel': 0, 'dct': 0, 'perc': 0, 'grad': 0}

        pbar = tqdm(data_loader, desc=f"Ep {epoch + 1}/{training_params['num_epochs']}")
        for pixel_values, _ in pbar:
            pixel_values = pixel_values.to(device)

            # 显存优化：使用 set_to_none=True 替代默认的 zero_grad()
            optimizer.zero_grad(set_to_none=True)

            # --- 开启混合精度上下文 ---
            with autocast():
                outputs = model(pixel_values)
                pixel_loss = outputs.loss

                mask = outputs.mask.bool()

                # 准备 Patch 用于计算损失
                with torch.no_grad():
                    patches = model.patchify(pixel_values)
                    target_patches = patches[mask]

                pred_patches = outputs.logits[mask]

                # --- A. DCT Loss ---
                if model_params["norm_pix_loss"]:
                    mean = target_patches.mean(dim=-1, keepdim=True)
                    var = target_patches.var(dim=-1, keepdim=True)
                    target_norm = (target_patches - mean) / (var + 1.e-6) ** .5
                else:
                    target_norm = target_patches

                pred_2d = pred_patches.reshape(-1, model_params["num_channels"], patch_size, patch_size)
                targ_2d = target_norm.reshape(-1, model_params["num_channels"], patch_size, patch_size)

                dct_val = torch.mean(
                    torch.abs(dct_layer(pred_2d) - dct_layer(targ_2d)) * dct_w.unsqueeze(0).unsqueeze(0))
                w_dct_loss = dct_loss_params["lambda_dct"] * dct_val

                # --- 准备全图重建 (用于感知和梯度损失) ---
                with torch.no_grad():
                    x_true = patches
                if model_params["norm_pix_loss"]:
                    mean = x_true.mean(dim=-1, keepdim=True)
                    var = x_true.var(dim=-1, keepdim=True)
                    std = (var + 1.e-6) ** .5
                    x_recon_patches = outputs.logits * std + mean
                else:
                    x_recon_patches = outputs.logits

                full_recon_patches = torch.where(mask.unsqueeze(-1), x_recon_patches, x_true)
                full_recon_img = model.unpatchify(full_recon_patches)

                # --- B. Perceptual Loss ---
                w_perc_loss = loss_weights["perceptual"] * perceptual_loss_func(full_recon_img, pixel_values)

                # --- C. Gradient Loss ---
                w_grad_loss = loss_weights["gradient"] * gradient_loss(full_recon_img, pixel_values)

                # 总损失
                loss = pixel_loss + w_dct_loss + w_perc_loss + w_grad_loss

            # --- 使用 Scaler 进行反向传播 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计 (需转回 float32 累加)
            avg['total'] += loss.item()
            avg['pixel'] += pixel_loss.item()
            avg['dct'] += w_dct_loss.item()
            avg['perc'] += w_perc_loss.item()
            avg['grad'] += w_grad_loss.item()

            pbar.set_postfix({
                'Tot': f"{loss.item():.3f}",
                'G': f"{w_grad_loss.item():.3f}",
                'P': f"{pixel_loss.item():.3f}"
            })

        # Epoch 结束：清理缓存
        torch.cuda.empty_cache()

        # 计算 Epoch 平均值
        for k in avg: avg[k] /= len(data_loader)

        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg['total'])
        history['pixel_loss'].append(avg['pixel'])
        history['dct_loss'].append(avg['dct'])
        history['perceptual_loss'].append(avg['perc'])
        history['grad_loss'].append(avg['grad'])

        print(f"Ep {epoch + 1} Stats: Total:{avg['total']:.4f} | Pix:{avg['pixel']:.4f} | DCT:{avg['dct']:.4f}")

        # 保存日志
        new_row = pd.DataFrame([{
            'Epoch': epoch + 1,
            'Total Loss': avg['total'], 'Pixel Loss': avg['pixel'],
            'Weighted DCT Loss': avg['dct'], 'Perceptual Loss': avg['perc'],
            'Gradient Loss': avg['grad']
        }])
        loss_df = pd.concat([loss_df, new_row], ignore_index=True)
        try:
            loss_df.to_excel(loss_log_path, index=False)
        except Exception as e:
            print(f"Save log failed: {e}")

        # 保存模型
        if avg['total'] < best_loss:
            best_loss = avg['total']
            print(">> New Best Model! Saving...")
            save_path = os.path.join(training_params["output_dir"], "best_model")
            model.save_pretrained(save_path)
            image_processor.save_pretrained(save_path)

        if not saving_params["save_best_only"]:
            curr_path = os.path.join(training_params["output_dir"], f"model_ep{epoch + 1}")
            model.save_pretrained(curr_path)
            image_processor.save_pretrained(curr_path)

            old_ep = epoch + 1 - saving_params["num_checkpoints_to_keep"]
            if old_ep > 0:
                old_path = os.path.join(training_params["output_dir"], f"model_ep{old_ep}")
                if os.path.exists(old_path):
                    shutil.rmtree(old_path, ignore_errors=True)

        # 可视化
        generate_detailed_visualizations(
            model, epoch, device, dataset.samples,
            image_processor, model_params, training_params["output_dir"]
        )

    # 训练结束
    plot_loss_curves(history, training_params["output_dir"])
    print("Training finished successfully.")


if __name__ == '__main__':
    main()