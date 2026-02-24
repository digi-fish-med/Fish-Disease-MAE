# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

# --- DCT 模块 ---
def create_dct_matrix(N):
    matrix = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                matrix[k, n] = 1.0 / np.sqrt(N)
            else:
                matrix[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * N))
    return torch.from_numpy(matrix).float()

class DCT2D(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        dct_mat = create_dct_matrix(size)
        self.register_buffer('dct_mat', dct_mat)
        self.register_buffer('dct_mat_t', dct_mat.t())

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.dct_mat @ x
        x = torch.transpose(self.dct_mat @ torch.transpose(x, -2, -1), -2, -1)
        return x

# --- VGG 感知损失模块 ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("Loading VGG19 for Perceptual Loss...")
        try:
            # 尝试在线加载
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except:
            print("Warning: Failed to load VGG weights online, trying local or default...")
            vgg = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18): self.slice3.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False
        self.to(device)
        self.eval()

    def forward(self, input, target):
        h_x = input
        h_y = target
        h1_x, h1_y = self.slice1(h_x), self.slice1(h_y)
        h2_x, h2_y = self.slice2(h1_x), self.slice2(h1_y)
        h3_x, h3_y = self.slice3(h2_x), self.slice3(h2_y)

        loss = torch.mean(torch.abs(h1_x - h1_y)) + \
               torch.mean(torch.abs(h2_x - h2_y)) + \
               torch.mean(torch.abs(h3_x - h3_y))
        return loss

# --- 梯度损失 (Gradient Smoothing Loss) ---
def gradient_loss(gen_frames, gt_frames):
    def get_gradients(x):
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        dx = r - l
        dy = t - b
        return dx, dy

    gen_dx, gen_dy = get_gradients(gen_frames)
    gt_dx, gt_dy = get_gradients(gt_frames)

    loss_dx = torch.abs(gen_dx - gt_dx)
    loss_dy = torch.abs(gen_dy - gt_dy)

    return torch.mean(loss_dx + loss_dy)