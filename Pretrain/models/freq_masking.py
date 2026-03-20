import torch
import torch.nn as nn
import random


class FreqR2MAEMasking(nn.Module):
    def __init__(self, p_min=0.1, p_max=0.3):
        """
        :param p_min: 最小掩码率 (当前设为 10%)
        :param p_max: 最大掩码率 (当前设为 50%)
        """
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max

    def generate_symmetric_mask(self, B, C, H, W, mask_ratio, device):
        """
        生成严格满足中心对称的频域随机掩码矩阵，修复偶数尺寸下的对称对齐问题
        假设输入尺寸为偶数 (如 224x224)
        """
        half_H = H // 2
        half_W = W // 2

        # 1. 随机生成上半平面的掩码 (对于 224，这里是 112 行)
        top_half = (torch.rand((B, 1, half_H, W), device=device) > mask_ratio).float()

        # 2. 生成下半平面：
        # 对于偶数尺寸，index 0 是 Nyquist 频率无对称项，因此翻转时剔除 index 0
        # (对于 224，这里变成了 111 行)
        bottom_half = torch.flip(top_half[:, :, 1:, :], dims=[2, 3])

        # 3. 处理中间行 (Nyquist & DC 行)，同样需要左右对称
        # 左半边 (112 列)
        middle_row_left = (torch.rand((B, 1, 1, half_W), device=device) > mask_ratio).float()
        # 右半边 (剔除 index 0，剩下 111 列)
        middle_row_right = torch.flip(middle_row_left[:, :, :, 1:], dims=[3])

        # 中心点(0,0)是直流分量(DC Component)，始终保留以维持图像基础亮度
        dc_point = torch.ones((B, 1, 1, 1), device=device)

        # 拼接中间行 (112 + 1 + 111 = 224 列)
        middle_row = torch.cat([middle_row_left, dc_point, middle_row_right], dim=3)

        # 4. 纵向拼凑出完整的掩码矩阵 (112 + 1 + 111 = 224 行)
        mask = torch.cat([top_half, middle_row, bottom_half], dim=2)

        return mask

    def forward(self, images):
        """
        images: 输入的 Batch 图像，shape [B, C, H, W]
        """
        B, C, H, W = images.shape
        device = images.device

        # 【核心 R^2 机制】：在这个 Mini-batch 动态采样一个随机掩码率
        current_mask_ratio = random.uniform(self.p_min, self.p_max)

        orig_dtype = images.dtype
        features_fp32 = images.to(torch.float32)

        # 1. FFT 转换到频域，并将低频(中心)移到矩阵正中央
        fft_img = torch.fft.fft2(features_fp32)
        fft_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

        # 2. 生成掩码
        mask = self.generate_symmetric_mask(B, C, H, W, current_mask_ratio, device)

        masked_fft_shifted = fft_shifted * mask



        # 5. 逆变换回空间域
        ifft_shifted = torch.fft.ifftshift(masked_fft_shifted, dim=(-2, -1))
        masked_images = torch.fft.ifft2(ifft_shifted).real

        # 因为掩码严格共轭对称，逆变换后虚部极小（机器精度误差），直接取实部
        masked_images = masked_images.to(orig_dtype)

        return masked_images, current_mask_ratio