# coding=utf-8
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_loss_curves(history, output_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(history['epoch'], history['total_loss'], label='Total Loss', linewidth=2, color='black')
    plt.plot(history['epoch'], history['pixel_loss'], '--', label='Pixel Loss (MSE)')
    plt.plot(history['epoch'], history['dct_loss'], ':', label='Weighted DCT Loss')
    plt.plot(history['epoch'], history['perceptual_loss'], '-.', label='Perceptual Loss (VGG)')
    plt.plot(history['epoch'], history['grad_loss'], '-', alpha=0.6, label='Gradient Loss')

    plt.title('Training Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()


def generate_detailed_visualizations(model, epoch, device, vis_dataset, image_processor, model_params, output_dir):
    """
    修改版：修复了 bool tensor 减法报错的问题
    """
    print(f"\nGenerating visualization for epoch {epoch + 1}...")
    model.eval()
    if len(vis_dataset) == 0:
        return

    # 随机选取一张图片
    img_idx = np.random.randint(0, len(vis_dataset))
    original_pil_image, _ = vis_dataset[img_idx]

    # 准备输入
    pixel_values_model = image_processor(images=original_pil_image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values_model)

        # 获取 Mask 和 重建 Patch
        y_pred_patches = outputs.logits
        x_true_patches = model.patchify(pixel_values_model)
        mask = outputs.mask.bool()

        # 反归一化/恢复 Patch
        mean = x_true_patches.mean(dim=-1, keepdim=True)
        var = x_true_patches.var(dim=-1, keepdim=True)
        std = (var + 1.e-6) ** .5
        x_recon_patches = y_pred_patches * std + mean

        # 拼接：未掩码部分使用原图 Patch，掩码部分使用重建 Patch
        reconstructed_patches = torch.where(mask.unsqueeze(-1), x_recon_patches, x_true_patches)
        reconstructed_image_tensor = model.unpatchify(reconstructed_patches)

        # 反归一化图像到 [0, 1] 用于显示
        global_mean = torch.tensor(image_processor.image_mean, device=device).view(1, -1, 1, 1)
        global_std = torch.tensor(image_processor.image_std, device=device).view(1, -1, 1, 1)

        recon_img = torch.clip((reconstructed_image_tensor * global_std) + global_mean, 0, 1).squeeze(0).cpu()

        # 处理原图用于显示
        vis_transform = transforms.Compose([
            transforms.Resize((model_params["image_size"], model_params["image_size"])),
            transforms.ToTensor(),
        ])
        orig_img = vis_transform(original_pil_image)

        # 生成掩码可视化图
        mask_vis = mask.detach().unsqueeze(-1).repeat(1, 1, model.config.patch_size ** 2 * 3)
        mask_vis = model.unpatchify(mask_vis).squeeze(0).cpu()

        # 将 mask_vis 强制转换为 float 类型后再进行计算
        masked_input_vis = orig_img * (1 - mask_vis.float())

        # 绘图：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Epoch {epoch + 1} Visualization', fontsize=16)

    # 1. Original
    axes[0].imshow(orig_img.permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 2. Masked Input
    axes[1].imshow(masked_input_vis.permute(1, 2, 0).numpy())
    axes[1].set_title("Masked Input")
    axes[1].axis('off')

    # 3. Reconstructed
    axes[2].imshow(recon_img.permute(1, 2, 0).numpy())
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"vis_epoch_{epoch + 1}.png"))
    plt.close(fig)