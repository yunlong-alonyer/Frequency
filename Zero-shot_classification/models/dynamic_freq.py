import torch
import torch.nn as nn


# =========================================================================
# 基于全量文本先验的 Master 频域门控 (Text-Driven Master Filter)
# =========================================================================
class TextDrivenGlobalFreqFilter(nn.Module):
    def __init__(self, text_dim=768, num_rings=32, base_retention=0.1):
        super().__init__()
        self.num_rings = num_rings
        self.base_retention = base_retention

        # 文本语义到 32 个频段权重的映射器
        self.text_to_rings = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_rings),
            nn.Sigmoid()
        )

    def forward(self, images, disease_book):
        B, C, H, W = images.shape
        device = images.device

        # 1. 喂入全部 75 种疾病向量，生成 75 组 32维权重
        # disease_book shape: (75, 768)
        all_weights = self.text_to_rings(disease_book.to(device))  # 形状: (75, 32)

        # 2. 核心大招：取所有疾病在各个频段的 "最大需求量" (Master Filter)
        master_weights, _ = torch.max(all_weights, dim=0)  # 形状: (32, )

        # 加上保底保留率，防止特征完全断裂
        master_weights = self.base_retention + (1.0 - self.base_retention) * master_weights

        # 3. 构建 2D 频域掩码画布
        half_H, half_W = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist = torch.sqrt((y - half_H) ** 2 + (x - half_W) ** 2)

        max_dist = torch.sqrt(torch.tensor(half_H ** 2 + half_W ** 2, device=device, dtype=torch.float32))
        ring_indices = (dist / max_dist * (self.num_rings - 1)).long()
        ring_indices = torch.clamp(ring_indices, 0, self.num_rings - 1)

        # 广播到图像维度 (1, 1, H, W)
        mask_2d = master_weights[ring_indices].unsqueeze(0).unsqueeze(0)

        # 4. 频域外科手术
        orig_dtype = images.dtype
        images_fp32 = images.to(torch.float32)

        fft_img = torch.fft.fft2(images_fp32)
        fft_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

        # 施加定制滤镜
        masked_fft = fft_shifted * mask_2d

        ifft_shifted = torch.fft.ifftshift(masked_fft, dim=(-2, -1))
        filtered_images = torch.fft.ifft2(ifft_shifted).real
        filtered_images = filtered_images.to(orig_dtype)

        return filtered_images